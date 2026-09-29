"""Reinforcement learning (GRPO) on a Tinker-compatible API: a fresh Qwen3.6-35B-A3B LoRA learns the reasoning_gym
tasks from the verifier reward alone. Thinking is on.

Every update the student samples `--group-size` responses to each of `--prompts` training problems (split equally
between `--domains`). The reward is the shared scorer's 0/1 score; each response's advantage is its reward normalized
within its prompt's group, applied to every response token. The update is one clipped PPO step (ratio 0.8-1.2).
Groups with no reward variance are kept with zero advantage. The settings are those of the released GRPO teachers
(hosted/tinker/README.md): 32 prompts x 8 responses, LR 1e-5 for LoRA, Adam betas 0.9/0.98, gradient clip 1.

    python hosted/opd/rl.py --domains caesar_cipher --output /abs/rl-caesar --backend skyrl
    python hosted/opd/rl.py --output /abs/rl-both --updates 20

Records, evaluation, checkpoints and --policy-lag behave as in opd.py and mopd.py (see common.py).
"""
import argparse
import json
import statistics
import time

import tinker

from common import Experiment, add_protocol_arguments, check_protocol_arguments, policy_datum, run
from shared import recipe, scoring

GRPO = {'prompts': 32, 'group_size': 8, 'clip_low_threshold': 0.8, 'clip_high_threshold': 1.2}
ADAM = {'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-8, 'weight_decay': 0.0, 'grad_clip_norm': 1.0}


def group_advantages(rewards):
    """Rewards normalized within one prompt's group; a group with one response or no variance gets zeros."""
    mean = statistics.mean(rewards)
    std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    return [(r - mean) / (std + 1e-6) for r in rewards]


class RL(Experiment):
    """An Experiment whose rollouts are groups scored by the verifier instead of by a teacher."""

    def start(self):
        a = self.args
        self.teachers = {}
        if a.resume:
            train = self.backend.resume_student(a.resume)
        else:
            train = self.backend.new_student(self.student_model[0], a.rank, a.seed, {'run': self.out.name, 'kind': self.kind})
        try:
            (self.out/'model-info.json').write_text(train.get_info().model_dump_json(indent=2))
        except Exception as error:
            (self.out/'model-info.json').write_text(json.dumps({'unavailable': str(error)[:300]}))
        return train, self.backend.sync(train, f'update-{a.start_update:03d}')[0]

    def rollout(self, student, domain, row, seed, t0):
        """One prompt's group. A multi-response request leaves the seed unset, as in the cookbook."""
        a = self.args
        prompt = self.prompt(row)
        begin = time.monotonic()
        result = student.sample(tinker.ModelInput.from_ints(prompt), num_samples=a.group_size,
                                sampling_params=self.params(1.0, seed if a.group_size == 1 else None)).result()
        sampled = time.monotonic()
        assert len(result.sequences) == a.group_size
        samples = []
        for j, sequence in enumerate(result.sequences):
            tokens, logprobs = list(sequence.tokens), list(sequence.logprobs)
            assert len(tokens) == len(logprobs)
            response = self.tokenizer.decode(tokens, skip_special_tokens=True)
            samples.append({'sample': j, 'tokens': tokens, 'logprobs': logprobs, 'response': response,
                            'score': scoring.score(response, row['label']), 'stop_reason': sequence.stop_reason})
        for s, advantage in zip(samples, group_advantages([s['score'] for s in samples])):
            s['advantage'] = advantage
        return {'domain': domain, 'index': row['_index'], 'seed': seed, 'prompt_tokens': prompt, 'samples': samples,
                'submitted': begin - t0, 'sampled': sampled - t0, 'scored': sampled - t0}

    def update(self, train, batch, prefetch):
        a = self.args
        update = batch['update']
        t0 = time.monotonic()
        self.log('update_start', update=update + 1, prompts=len(batch['futures']))
        groups = [f.result() for f in batch['futures']]
        rollout_seconds = time.monotonic() - t0
        lag = update - batch['version']
        assert 0 <= lag <= a.policy_lag, 'Refuse a stale or future-policy batch'
        upcoming = prefetch()
        self.save_rollouts(f'update-{update+1:03d}', [{'update': update + 1, 'policy_lag': lag, **g} for g in groups])

        train_start = time.monotonic()
        flat = [(g, s) for g in groups for s in g['samples']]
        datums = [policy_datum(g['prompt_tokens'], s['tokens'], s['logprobs'], [s['advantage']] * len(s['tokens']))
                  for g, s in flat]
        fb = train.forward_backward(datums, loss_fn='ppo', loss_fn_config={
            'clip_low_threshold': a.clip_low, 'clip_high_threshold': a.clip_high})
        opt = train.optim_step(tinker.AdamParams(learning_rate=a.learning_rate, **ADAM))
        fb, opt = fb.result(), opt.result()
        train_seconds = time.monotonic() - train_start
        mismatch = [float(x) - y for (g, s), o in zip(flat, fb.loss_fn_outputs, strict=True) if 'logprobs' in o
                    for x, y in zip(o['logprobs'].data[len(g['prompt_tokens']) - 1:], s['logprobs'], strict=True)]

        sync_start = time.monotonic()
        student, sampler_path = self.backend.sync(train, f'update-{update+1:03d}')
        sync_seconds = time.monotonic() - sync_start
        self.append('metrics.jsonl', self.rl_metrics(update, groups, t0, rollout_seconds, train_seconds, sync_seconds,
                                                     mismatch, sampler_path, fb, opt, lag))
        return student, upcoming

    def rl_metrics(self, update, groups, t0, rollout_seconds, train_seconds, sync_seconds, mismatch, sampler_path,
                   fb, opt, lag):
        samples = [s for g in groups for s in g['samples']]
        sampled = [g['sampled'] for g in groups]
        response_tokens = sum(len(s['tokens']) for s in samples)
        prompt_tokens = sum(len(g['prompt_tokens']) * len(g['samples']) for g in groups)
        by_domain = {}
        for d in self.domains:
            gs = [g for g in groups if g['domain'] == d]
            ss = [s for g in gs for s in g['samples']]
            by_domain[d] = {'groups': len(gs), 'responses': len(ss), 'reward': statistics.mean(s['score'] for s in ss),
                            'groups_with_signal': sum(len({s['score'] for s in g['samples']}) > 1 for g in gs),
                            'mean_response_tokens': statistics.mean(len(s['tokens']) for s in ss)}
        update_seconds = time.monotonic() - t0
        metric = {
            'update': update + 1, 'policy_lag': lag, 'groups': len(groups), 'responses': len(samples),
            'learning_rate': self.args.learning_rate, 'prompt_tokens': prompt_tokens, 'response_tokens': response_tokens,
            'max_response_tokens': max(len(s['tokens']) for s in samples),
            'truncated': sum(s['stop_reason'] == 'length' for s in samples), 'by_domain': by_domain,
            'reward': statistics.mean(s['score'] for s in samples),
            'nonzero_advantage_responses': sum(s['advantage'] != 0 for s in samples),
            'mean_unique_responses_per_group': statistics.mean(len({s['response'] for s in g['samples']}) for g in groups),
            'timing': {'rollout': rollout_seconds, 'train': train_seconds, 'sync': sync_seconds, 'update': update_seconds,
                       'first_group': min(sampled), 'median_group': statistics.median(sampled),
                       'last_group': max(sampled)},
            'throughput': {'sampled_tokens_per_second': response_tokens / max(sampled),
                           'trained_tokens_per_second': (prompt_tokens + response_tokens) / train_seconds,
                           'update_response_tokens_per_second': response_tokens / update_seconds},
            'train_sampler_logprob_delta_mean': statistics.mean(mismatch) if mismatch else None,
            'train_sampler_logprob_delta_abs_mean': statistics.mean(abs(x) for x in mismatch) if mismatch else None,
            'sampler_path': sampler_path, 'forward_backward_metrics': fb.metrics, 'optimizer_metrics': opt.metrics}
        self.log('update_complete', **{k: v for k, v in metric.items() if k != 'forward_backward_metrics'})
        return metric


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--domains', choices=recipe.DOMAINS, nargs='+', default=list(recipe.DOMAINS))
    parser.add_argument('--group-size', type=int, default=GRPO['group_size'])
    parser.add_argument('--clip-low', type=float, default=GRPO['clip_low_threshold'])
    parser.add_argument('--clip-high', type=float, default=GRPO['clip_high_threshold'])
    add_protocol_arguments(parser)
    parser.set_defaults(prompts=GRPO['prompts'])
    args = parser.parse_args()
    check_protocol_arguments(parser, args)
    if args.group_size < 2: parser.error('--group-size must be at least 2 for group-normalized advantages')
    if args.prompts % len(args.domains): parser.error('--prompts must split evenly over the domains')
    domains = [d for d in recipe.DOMAINS if d in args.domains]
    share = args.prompts // len(domains)

    class Run(RL):
        def record_inputs(self):
            super().record_inputs()
            config = json.loads((self.out/'config.json').read_text())
            config.update(loss='ppo', advantage='group-normalized verifier reward', adam=ADAM,
                          clip=[args.clip_low, args.clip_high], teachers=None)
            (self.out/'config.json').write_text(json.dumps(config, indent=2))

    run(lambda: Run(args, 'rl', {d: None for d in domains}),
        lambda experiment, update: [p for d in domains for p in experiment.train_prompts(d, update, share)])


if __name__ == '__main__':
    main()
