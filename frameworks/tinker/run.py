"""Remote puzzle GRPO and routed sampled-token OPD using the Tinker SDK.

Adapted from the first-party cookbook's recipes/rl_loop.py and
distillation/train_on_policy.py. No model weights are downloaded.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

import tinker
from transformers import AutoTokenizer
from tinker_cookbook import renderers

ROOT = Path(__file__).resolve().parents[2]
MODEL = 'Qwen/Qwen3.6-35B-A3B'
REVISION = '995ad96eacd98c81ed38be0c5b274b04031597b0'
DOMAINS = ('countdown', 'graph_color')
spec = importlib.util.spec_from_file_location('puzzle_scoring', ROOT/'frameworks/slime/campaign/scoring.py')
scoring = importlib.util.module_from_spec(spec)
# Keep bytecode out of the archived campaign-template directory.
_prior_bytecode_setting = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    spec.loader.exec_module(scoring)
finally:
    sys.dont_write_bytecode = _prior_bytecode_setting


def load_data(domain, split):
    name = ('countdown4' if domain == 'countdown' else 'graph12') + f'-{split}.jsonl'
    raw = gzip.decompress((ROOT/'data'/f'{name}.gz').read_bytes())
    manifest = json.loads((ROOT/'data/manifest.json').read_text())[name]
    assert hashlib.sha256(raw).hexdigest() == manifest['sha256']
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == manifest['rows']
    for i, row in enumerate(rows):
        assert row['metadata']['domain'] == domain
        row['_index'] = i
    return rows


def centered_advantages(rewards):
    # Match verifier GRPO: normalize within each prompt's response group.
    mean = statistics.mean(rewards)
    std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    return [(v - mean) / (std + 1e-6) for v in rewards]


def routed_advantages(prompt_length, response_logprobs, full_teacher_logprobs):
    """Teacher logprob index j predicts full_sequence[j], including prompt slots."""
    teacher = full_teacher_logprobs[prompt_length:]
    assert len(teacher) == len(response_logprobs)
    assert all(v is not None and math.isfinite(v) for v in teacher)
    return teacher, [t-s for t,s in zip(teacher, response_logprobs)]


def datum(prompt_tokens, response_tokens, logprobs, advantages):
    assert response_tokens and len(response_tokens) == len(logprobs) == len(advantages)
    assert all(math.isfinite(x) for x in logprobs + advantages)
    prefix = len(prompt_tokens) - 1
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(prompt_tokens + response_tokens[:-1]),
        loss_fn_inputs={
            'target_tokens': tinker.TensorData(data=[0] * prefix + response_tokens, dtype='int64', shape=[prefix + len(response_tokens)]),
            'logprobs': tinker.TensorData(data=[0.0] * prefix + logprobs, dtype='float32', shape=[prefix + len(response_tokens)]),
            'advantages': tinker.TensorData(data=[0.0] * prefix + advantages, dtype='float32', shape=[prefix + len(response_tokens)]),
        },
    )


class Run:
    def __init__(self, args):
        self.args = args
        self.out = args.output.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.pool = ThreadPoolExecutor(max_workers=args.concurrency)
        self.event('start', config={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
        (self.out/'config.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, indent=2))
        (self.out/'versions.json').write_text(json.dumps({n: importlib.metadata.version(n) for n in ('tinker', 'tinker-cookbook', 'transformers', 'torch', 'numpy')}, indent=2))
        (self.out/'source.py').write_text(Path(__file__).read_text())
        (self.out/'scoring.py').write_text((ROOT/'frameworks/slime/campaign/scoring.py').read_text())
        (self.out/'data-manifest.json').write_text((ROOT/'data/manifest.json').read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.renderer = renderers.get_renderer('qwen3_5_disable_thinking', self.tokenizer, model_name=MODEL)
        self.stop = self.renderer.get_stop_sequences()
        self.service = tinker.ServiceClient(user_metadata={'experiment': 'opd-puzzles-20260921', 'stage': args.mode})
        caps = self.service.get_server_capabilities()
        assert MODEL in [m.model_name for m in caps.supported_models]
        (self.out/'capabilities.json').write_text(caps.model_dump_json(indent=2))
        self.rows = {d: {s: load_data(d, s) for s in ('train', 'dev')} for d in DOMAINS}
        self.order = {}
        for d in DOMAINS:
            self.order[d] = list(range(len(self.rows[d]['train'])))
            random.Random(args.seed + DOMAINS.index(d)).shuffle(self.order[d])
        self.event('ready', stop=self.stop, tokenizer_revision=REVISION)

    def write(self, filename, row):
        with (self.out/filename).open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')

    def event(self, name, **data):
        row = {'event': name, 'utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-self.started, **data}
        self.write('events.jsonl', row)
        print(json.dumps(row), flush=True)

    def prompt(self, row):
        # Use the pinned original HF template; fail if cookbook rendering differs.
        ids = self.tokenizer.apply_chat_template(row['prompt'], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
        cooked = self.renderer.build_generation_prompt(row['prompt']).to_ints()
        assert ids == cooked, 'Pinned HF and cookbook prompt tokens differ'
        assert len(ids) + self.args.max_tokens <= 2048
        return ids

    def sample_one(self, client, domain, row, group, temperature, seed):
        tokens = self.prompt(row)
        start = time.monotonic()
        result = client.sample(tinker.ModelInput.from_ints(tokens), num_samples=group,
            sampling_params=tinker.SamplingParams(max_tokens=self.args.max_tokens,
                temperature=temperature, top_p=1.0, top_k=-1, stop=self.stop,
                seed=seed if group == 1 else None)).result()
        assert len(result.sequences) == group
        records = []
        for j, seq in enumerate(result.sequences):
            response = self.tokenizer.decode(seq.tokens, skip_special_tokens=True)
            assert seq.logprobs is not None and len(seq.tokens) == len(seq.logprobs)
            records.append({'domain': domain, 'index': row['_index'], 'sample': j,
                'prompt_tokens': tokens, 'tokens': seq.tokens, 'logprobs': seq.logprobs,
                'response': response, 'score': scoring.score(response, row['label']),
                'stop_reason': seq.stop_reason, 'request_seconds': time.monotonic()-start,
                'seed': seed if group == 1 else None})
        return records

    def sample_batch(self, client, work, group, temperature, step, split):
        futures = {self.pool.submit(self.sample_one, client, d, row, group, temperature,
                    self.args.seed + step*100000 + i): i for i, (d,row) in enumerate(work)}
        gathered = {}
        for f in as_completed(futures):
            i = futures[f]
            records = f.result()
            for r in records:
                self.write('samples.jsonl', {'step': step, 'split': split, **r})
            gathered[i] = records
        return [gathered[i] for i in range(len(work))]

    def evaluate(self, client, step):
        start = time.monotonic()
        work = [(d,r) for d in DOMAINS for r in self.rows[d]['dev'][:self.args.eval_size]]
        groups = self.sample_batch(client, work, 1, 0.0, step, 'dev')
        metrics = {}
        for d in DOMAINS:
            rs = [r for g in groups for r in g if r['domain'] == d]
            n = len(rs); correct = sum(r['score'] for r in rs)
            metrics[d] = {'n': n, 'correct': correct, 'accuracy': correct/n,
                'mean_tokens': statistics.mean(len(r['tokens']) for r in rs),
                'truncated': sum(r['stop_reason'] == 'length' for r in rs)}
        self.write('evaluations.jsonl', {'step': step, 'seconds': time.monotonic()-start, **metrics})
        self.event('evaluation', step=step, metrics=metrics)
        return metrics

    def checkpoint(self, train, step):
        name = f'step-{step:03d}'
        state = train.save_state(name=name).result()
        sampler = train.save_weights_for_sampler(name=name).result()
        row = {'step': step, 'state_path': state.path, 'sampler_path': sampler.path}
        self.write('checkpoints.jsonl', row)
        self.event('checkpoint', **row)
        return row

    def execute(self):
        a = self.args
        if a.mode == 'eval':
            client = self.service.create_sampling_client(model_path=a.checkpoint) if a.checkpoint else self.service.create_sampling_client(base_model=MODEL)
            self.evaluate(client, 0)
            self.event('complete')
            return
        if a.checkpoint:
            train = self.service.create_training_client_from_state_with_optimizer(a.checkpoint)
        else:
            train = self.service.create_lora_training_client(MODEL, rank=a.rank, seed=a.seed,
                user_metadata={'domain': a.domain or 'mixed', 'run': self.out.name})
        (self.out/'model-info.json').write_text(train.get_info().model_dump_json(indent=2))
        self.event('training_client_created')
        teachers = {}
        if a.mode == 'mopd':
            routes = json.loads(a.teachers.read_text())
            assert set(routes) == set(DOMAINS)
            (self.out/'teachers.json').write_text(json.dumps(routes, indent=2))
            teachers = {d: self.service.create_sampling_client(model_path=routes[d]) for d in DOMAINS}
        sample = train.save_weights_and_get_sampling_client()
        if not a.skip_baseline:
            self.evaluate(sample, a.start_step)
        completed_updates = 0
        completion_reason = 'step_budget_reached'
        for step in range(a.start_step, a.steps):
            start = time.monotonic()
            domains = (a.domain,) if a.mode == 'teacher' else DOMAINS
            count = a.prompts if a.mode == 'teacher' else a.prompts//2
            work = [(d,self.rows[d]['train'][self.order[d][(step*count+i)%len(self.order[d])]]) for d in domains for i in range(count)]
            self.event('step_start', step=step+1, prompts=len(work))
            groups = self.sample_batch(sample, work, a.group_size, 1.0, step, 'train')
            sampling_seconds = time.monotonic()-start
            flat = [r for g in groups for r in g]
            teacher_start = time.monotonic()
            if a.mode == 'teacher':
                for g in groups:
                    for r, advantage in zip(g, centered_advantages([r['score'] for r in g])):
                        r['advantages'] = [advantage] * len(r['tokens'])
            else:
                def teacher_score(r):
                    lp = teachers[r['domain']].compute_logprobs(tinker.ModelInput.from_ints(r['prompt_tokens']+r['tokens'])).result()
                    return routed_advantages(len(r['prompt_tokens']), r['logprobs'], lp)
                futures = {self.pool.submit(teacher_score,r): r for r in flat}
                for f in as_completed(futures):
                    r = futures[f]
                    r['teacher_logprobs'], r['advantages'] = f.result()
                    self.write('teacher-scores.jsonl', {'step': step, **r})
            teacher_seconds = time.monotonic()-teacher_start
            datums = [datum(r['prompt_tokens'],r['tokens'],r['logprobs'],r['advantages']) for r in flat]
            self.event('forward_backward_start', step=step+1, samples=len(datums))
            learn_start = time.monotonic()
            # Never application-retry a mutation: an ambiguous failure stops the run.
            fb_future = train.forward_backward(datums,
                loss_fn='ppo' if a.mode == 'teacher' else 'importance_sampling',
                loss_fn_config={'clip_low_threshold': 0.8, 'clip_high_threshold': 1.2} if a.mode == 'teacher' else None)
            opt_future = train.optim_step(tinker.AdamParams(learning_rate=a.learning_rate,
                beta1=0.9, beta2=0.98 if a.mode == 'teacher' else 0.999, eps=1e-8,
                weight_decay=0.1 if a.mode == 'teacher' else 0.0, grad_clip_norm=1.0))
            fb = fb_future.result()
            opt = opt_future.result()
            response_deltas = []
            for r, output in zip(flat, fb.loss_fn_outputs, strict=True):
                learner_lp = output['logprobs'].data[len(r['prompt_tokens'])-1:]
                assert len(learner_lp) == len(r['tokens'])
                differences = [float(x)-y for x,y in zip(learner_lp,r['logprobs'])]
                assert all(math.isfinite(x) for x in differences)
                response_deltas.extend(differences)
                self.write('learner-scores.jsonl', {'step': step, 'domain': r['domain'],
                    'index': r['index'], 'sample': r['sample'], 'logprobs': learner_lp,
                    'advantages': r['advantages']})
            learn_seconds = time.monotonic()-learn_start
            sample = train.save_weights_and_get_sampling_client()
            row = {'step': step+1, 'samples': len(flat), 'learning_rate': a.learning_rate,
                'sampling_seconds': sampling_seconds, 'teacher_seconds': teacher_seconds,
                'train_seconds': learn_seconds, 'step_seconds': time.monotonic()-start,
                'prompt_tokens': sum(len(r['prompt_tokens']) for r in flat),
                'response_tokens': sum(len(r['tokens']) for r in flat),
                'response_train_sample_logprob_delta_mean': statistics.mean(response_deltas),
                'response_train_sample_logprob_delta_abs_mean': statistics.mean(abs(x) for x in response_deltas),
                'mean_unique_responses_per_group': statistics.mean(len(set(r['response'] for r in g)) for g in groups),
                'scores': {d: statistics.mean(r['score'] for r in flat if r['domain']==d) for d in domains},
                'nonzero_advantage_samples': sum(any(v != 0 for v in r['advantages']) for r in flat),
                'forward_backward_metrics': fb.metrics, 'optimizer_metrics': opt.metrics}
            if a.mode == 'mopd':
                row['sampled_reverse_kl_by_domain'] = {
                    d: -statistics.mean(x for r in flat if r['domain'] == d for x in r['advantages'])
                    for d in domains}
                row['accepted_policy_lag'] = 0
            self.write('metrics.jsonl', row)
            self.event('step_complete', **row)
            completed_updates += 1
            if (step+1)%a.save_every == 0 or step+1 == a.steps:
                self.checkpoint(train,step+1)
            if (step+1)%a.eval_every == 0 or step+1 == a.steps:
                evaluation = self.evaluate(sample,step+1)
                if a.mode == 'teacher' and a.target_score is not None and evaluation[a.domain]['accuracy'] >= a.target_score:
                    if (step+1)%a.save_every != 0 and step+1 != a.steps:
                        self.checkpoint(train,step+1)
                    completion_reason = 'teacher_target_reached'
                    break
        self.event('complete', optimizer_updates=completed_updates,
            total_optimizer_updates=a.start_step+completed_updates, reason=completion_reason)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('eval','teacher','mopd'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--domain',choices=DOMAINS)
    p.add_argument('--teachers',type=Path)
    p.add_argument('--checkpoint')
    p.add_argument('--start-step',type=int,default=0)
    p.add_argument('--steps',type=int,default=40)
    p.add_argument('--prompts',type=int)
    p.add_argument('--group-size',type=int)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--target-score',type=float,help='Optional teacher-only dev threshold; stops after a scheduled evaluation')
    p.add_argument('--rank',type=int,default=64)
    p.add_argument('--seed',type=int,default=20260921)
    p.add_argument('--max-tokens',type=int,default=256)
    p.add_argument('--eval-size',type=int,default=512)
    p.add_argument('--eval-every',type=int,default=10)
    p.add_argument('--save-every',type=int,default=10)
    p.add_argument('--concurrency',type=int,default=32)
    p.add_argument('--skip-baseline',action='store_true')
    args = p.parse_args()
    args.prompts = args.prompts or (32 if args.mode == 'teacher' else 128)
    args.group_size = args.group_size or (8 if args.mode == 'teacher' else 1)
    for key in ('prompts', 'group_size', 'max_tokens', 'rank', 'eval_every', 'save_every', 'concurrency'):
        if getattr(args, key) <= 0: p.error(f'{key} must be positive')
    if not 1 <= args.eval_size <= 512: p.error('--eval-size must be between 1 and 512')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0: p.error('learning rate must be finite and positive')
    if args.mode != 'eval' and not 0 <= args.start_step < args.steps: p.error('require 0 <= start-step < steps')
    if args.start_step and not args.checkpoint: p.error('--start-step requires an optimizer-state checkpoint')
    if args.mode == 'teacher' and not args.domain: p.error('teacher requires --domain')
    if args.target_score is not None and (args.mode != 'teacher' or not 0 < args.target_score <= 1):
        p.error('--target-score is a teacher-only fraction in (0, 1]')
    if args.mode == 'mopd' and (not args.teachers or args.prompts%2 or args.group_size != 1): p.error('MOPD requires --teachers, even prompt count, and group size 1')
    run = None
    try:
        run = Run(args)
        run.execute()
    except BaseException as exc:
        if run:
            run.event('failed', error_type=type(exc).__name__, message=str(exc))
            (run.out/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        if run: run.pool.shutdown(wait=False, cancel_futures=True)


if __name__ == '__main__':
    main()
