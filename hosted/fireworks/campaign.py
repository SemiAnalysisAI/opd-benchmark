"""Hosted teacher RL, then routed multi-teacher OPD, on the Fireworks dedicated Training API. Thinking is on.

    python campaign.py --root NEW_RUN_ROOT --tokenizer-info TOKENIZER_JSON --account ACCOUNT
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse, importlib.metadata, json, logging, math, os, random, shutil, statistics, subprocess, sys, time
import traceback

import tinker
from transformers import AutoTokenizer
from fireworks.training.sdk import FiretitanServiceClient, TrainerJobManager, TrainerJobConfig
from fireworks.training.sdk.concurrency import FixedConcurrencyController

from lifecycle import (  # Also makes the shared package importable.
    CLEANUP_RESERVE_USD, DEFAULT_BUDGET_USD, RATE_USD_HOUR, REPO, api_key, check, cleanup, event, read_json, utc_now,
    verify_account, write)
from shared import credentials, puzzles, scoring
from shared.recipe import (CONTEXT_LENGTH, DOMAINS, ENABLE_THINKING, EVAL_EXAMPLES, EVAL_INTERVAL, LEARNING_RATE,
                           MAX_RESPONSE_TOKENS, PROMPTS_PER_UPDATE, SAMPLES_PER_PROMPT, UPDATES)

MODEL = 'accounts/fireworks/models/qwen3p5-35b-a3b'
SHAPE = 'accounts/fireworks/trainingShapes/qwen3p5-35b-a3b-256k-lora/versions/e3dzebwt'
LORA_RANK = 64
SEED = 20260921
SEED_STRIDE = 100_000  # Sampling seed: SEED + step * SEED_STRIDE + request index.
CONCURRENCY = 32
# Teacher GRPO from research/rl-teachers-2026-09-24/rl.template.toml: 32 prompts x 8 responses, at most
# 200 updates, evaluation every 25, no weight decay, LR 1e-6 there (10x for LoRA here).
TEACHER_DEFAULTS = {'prompts': 32, 'learning_rate': 1e-5, 'eval_every': 25, 'max_steps': 200, 'target_tolerance': 0}
TEACHER_GROUP_SIZE = 8
# Operational greedy dev-accuracy targets, just below the research teachers' held-out scores
# (caesar_cipher 73.3%, simple_geometry 99.8%); not a verified match of recipe.TEACHERS.
TEACHER_TARGETS = {'caesar_cipher': 0.70, 'simple_geometry': 0.95}
PPO_CLIP = {'clip_low_threshold': 0.8, 'clip_high_threshold': 1.2}
TEACHER_ADAM = {'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-8, 'weight_decay': 0.0, 'grad_clip_norm': 1.0}
STUDENT_LEARNING_RATE = 10 * LEARNING_RATE  # The recipe's full-finetuning LR, times 10 for LoRA.
STUDENT_ADAM = {**TEACHER_ADAM, 'beta2': 0.999, 'weight_decay': 0.0}
SCORING_NOTE = 'single-session checkpoint swaps; frozen teacher forwards; exact student optimizer restore'


def append(path, row):
    with Path(path).open('a') as file:
        file.write(json.dumps(row, allow_nan=False, default=str) + '\n')


def tensor(values, dtype):
    return tinker.TensorData(data=values, dtype=dtype, shape=[len(values)])


def scoring_datum(r):
    """Forward-only datum whose output log-probabilities score the sampled response."""
    full = r['prompt_tokens'] + r['tokens']
    weights = [0.0] * (len(r['prompt_tokens']) - 1) + [1.0] * len(r['tokens'])
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(full[:-1]),
                        loss_fn_inputs={'target_tokens': tensor(full[1:], 'int64'), 'weights': tensor(weights, 'float32')})


def training_datum(r):
    """Policy-gradient datum; prompt positions carry zero advantage."""
    pad = len(r['prompt_tokens']) - 1
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(r['prompt_tokens'] + r['tokens'][:-1]), loss_fn_inputs={
        'target_tokens': tensor([0] * pad + r['tokens'], 'int64'),
        'logprobs': tensor([0.0] * pad + r['logprobs'], 'float32'),
        'advantages': tensor([0.0] * pad + r['advantages'], 'float32')})


def response_logprobs(r, output):
    values = [float(x) for x in output['logprobs'].data[len(r['prompt_tokens']) - 1:]]
    assert len(values) == len(r['tokens']) and all(math.isfinite(x) for x in values)
    return values


class Campaign:
    def __init__(self, root, tokenizer, rows, teacher_settings=None, resume_plan=None):
        self.root, self.tokenizer, self.data = root, tokenizer, rows
        self.teacher_settings = s = {**TEACHER_DEFAULTS, **(teacher_settings or {})}
        assert set(s) == set(TEACHER_DEFAULTS) and isinstance(s['prompts'], int) and s['prompts'] > 0
        assert 0 < s['learning_rate'] < 1 and s['eval_every'] > 0 and s['max_steps'] > 0
        assert 0 <= s['target_tolerance'] < min(TEACHER_TARGETS.values())
        self.resume_plan = resume_plan or {}
        self.order = {d: list(range(len(rows[d]['train']))) for d in DOMAINS}  # Fixed shuffled training order.
        for d in DOMAINS:
            random.Random(SEED + DOMAINS.index(d)).shuffle(self.order[d])
        self.started = time.monotonic()
        self.pool = ThreadPoolExecutor(max_workers=CONCURRENCY)
        self.prompts, self.selected = {}, {}
        self.service = self.sampler = self.train = None

    def render_prompts(self):  # Before paying for GPUs.
        for d in DOMAINS:
            for split in ('train', 'dev'):
                for row in self.data[d][split]:
                    ids = self.tokenizer.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True,
                                                             enable_thinking=ENABLE_THINKING, return_dict=False)
                    assert len(ids) + MAX_RESPONSE_TOKENS <= CONTEXT_LENGTH
                    self.prompts[d, split, row['_index']] = ids

    def log(self, event_name, **fields):
        row = {'event': event_name, 'utc': utc_now(), 'elapsed_seconds': time.monotonic() - self.started, **fields}
        append(self.root / 'events.jsonl', row)
        print(json.dumps(row, default=str), flush=True)

    def publish(self, name):  # Snapshot the weights for sampling and point the sampler at them.
        check(self.root)
        start = time.monotonic()
        path = self.train.save_weights_for_sampler(name).result().path
        if self.sampler is not None:
            self.sampler.close()
            self.sampler = None
        self.sampler = self.service.create_sampling_client(
            model_path=path, tokenizer=self.tokenizer, training_client=self.train,
            concurrency_controller=FixedConcurrencyController(CONCURRENCY))
        self.log('published', path=path, name=name, seconds=time.monotonic() - start)
        return path

    def sample_one(self, domain, row, split, group, temperature, seed):
        check(self.root)
        ids = self.prompts[domain, split, row['_index']]
        start = time.monotonic()
        seed = seed if group == 1 else None  # A group needs distinct samples.
        params = tinker.SamplingParams(max_tokens=MAX_RESPONSE_TOKENS, temperature=temperature, top_p=1, top_k=-1,
                                       stop=[self.tokenizer.eos_token_id], seed=seed)
        result = self.sampler.sample(tinker.ModelInput.from_ints(ids), num_samples=group, sampling_params=params).result()
        records = []
        for sample, sequence in enumerate(result.sequences):
            response = self.tokenizer.decode(list(sequence.tokens), skip_special_tokens=True)
            records.append({'domain': domain, 'index': row['_index'], 'sample': sample, 'prompt_tokens': ids,
                            'tokens': list(sequence.tokens), 'logprobs': list(sequence.logprobs), 'response': response,
                            'score': scoring.score(response, row['label']), 'stop_reason': sequence.stop_reason,
                            'request_seconds': time.monotonic() - start, 'seed': seed})
        return records

    def sample_batch(self, out, work, step, split, group=1, temperature=1.0):  # Groups return in `work` order.
        check(self.root)
        futures = {self.pool.submit(self.sample_one, domain, row, split, group, temperature,
                                    SEED + step * SEED_STRIDE + i): i for i, (domain, row) in enumerate(work)}
        groups = {}
        for future in as_completed(futures):
            groups[futures[future]] = future.result()
            for record in groups[futures[future]]:
                append(out / 'samples.jsonl', {'step': step, 'split': split, **record})
        return [groups[i] for i in range(len(work))]

    def evaluate(self, out, step):  # Greedy accuracy on every domain's full dev split.
        start = time.monotonic()
        groups = self.sample_batch(out, [(d, row) for d in DOMAINS for row in self.data[d]['dev']], step, 'dev', 1, 0)
        result = {'step': step, 'seconds': time.monotonic() - start}
        for d in DOMAINS:
            rows = [r for group in groups for r in group if r['domain'] == d]
            result[d] = {'n': len(rows), 'correct': sum(r['score'] for r in rows),
                         'accuracy': statistics.mean(r['score'] for r in rows),
                         'mean_tokens': statistics.mean(len(r['tokens']) for r in rows),
                         'truncated': sum(r['stop_reason'] == 'length' for r in rows)}
        append(out / 'evaluations.jsonl', result)
        self.log('evaluation', run=out.name, **result)
        return result

    def score_with_teachers(self, out, step, records):
        """MOPD advantages: teacher minus student token log-probabilities. The one session holds the
        student, so save it with its optimizer, score each domain with its frozen teacher checkpoint,
        and restore the student before learning. Sampler snapshots are immutable meanwhile."""
        student_state = self.train.save_state(f'student-pre-{step:03d}').result().path
        self.log('student_state_saved', step=step, path=student_state)
        for d in DOMAINS:
            rows = [r for r in records if r['domain'] == d]
            teacher_state = self.selected[d]['checkpoint']['state_path']
            self.train.load_state(teacher_state).result()
            self.log('teacher_state_loaded', step=step, domain=d, path=teacher_state)
            result = self.train.forward([scoring_datum(r) for r in rows], loss_fn='cross_entropy').result()
            append(out / 'teacher-forward-metrics.jsonl', {'step': step, 'domain': d, 'metrics': result.metrics})
            for r, output in zip(rows, result.loss_fn_outputs, strict=True):
                r['teacher_logprobs'] = response_logprobs(r, output)
                r['advantages'] = [t - s for t, s in zip(r['teacher_logprobs'], r['logprobs'])]
                append(out / 'teacher-scores.jsonl', {'step': step, **r})
        self.train.load_state_with_optimizer(student_state).result()
        self.log('student_state_restored', step=step, path=student_state)

    def train_stage(self, domain=None):
        """Train one teacher (`domain` given) or the student (`domain` None)."""
        teacher = domain is not None
        name = 'teacher-' + domain if teacher else 'student'
        out = self.root / name
        out.mkdir()
        start = time.monotonic()
        s = self.teacher_settings
        resume = self.resume_plan.get(domain, {}) if teacher else {}
        domains = (domain,) if teacher else DOMAINS
        per_domain = s['prompts'] if teacher else PROMPTS_PER_UPDATE // len(DOMAINS)
        start_step, steps = resume.get('step', 0), s['max_steps'] if teacher else UPDATES
        # A resume plan carries prompt_offset when the teacher batch size changed.
        cfg = {'start_step': start_step, 'resume': resume, 'mode': 'teacher' if teacher else 'mopd', 'domain': domain,
               'steps': steps, 'group_size': TEACHER_GROUP_SIZE if teacher else SAMPLES_PER_PROMPT,
               'prompts': per_domain * len(domains), 'prompt_offset': resume.get('prompt_offset', start_step * per_domain),
               'rank': LORA_RANK, 'learning_rate': s['learning_rate'] if teacher else STUDENT_LEARNING_RATE,
               'seed': SEED, 'max_tokens': MAX_RESPONSE_TOKENS, 'eval_examples': EVAL_EXAMPLES,
               'eval_every': s['eval_every'] if teacher else EVAL_INTERVAL, 'target': TEACHER_TARGETS.get(domain),
               'stopping_threshold': TEACHER_TARGETS[domain] - s['target_tolerance'] if teacher else None,
               'fresh_base': not resume, 'teacher_scoring': SCORING_NOTE, 'pipeline_depth': 0, 'max_policy_lag': 0}
        assert 0 <= start_step <= steps
        write(out / 'config.json', cfg)
        best = resume.get('best')
        if resume.get('complete'):  # Reuse a finished teacher without touching the session.
            self.selected[domain] = {**best, 'source_root': resume['source_root']}
            write(self.root / 'teacher-selection.json', self.selected)
            write(out / 'complete.json', {'updates': start_step, 'reused': True, 'source_root': resume['source_root']})
            self.log('stage_restored', run=name, updates=start_step)
            return
        # The dedicated backend permits only one resident LoRA session. Reuse it,
        # restoring a pristine base weights/optimizer checkpoint for fresh stages.
        if self.train is None:
            self.train = self.service.create_training_client(
                base_model=MODEL, lora_rank=LORA_RANK, seed=SEED,
                user_metadata={'experiment': self.root.name, 'stage': 'shared-session'})
            self.initial_state = self.train.save_state('initial-base').result().path
            write(self.root / 'initial-state.json', {'path': self.initial_state})
        self.train.load_state_with_optimizer(resume['restore_reference'] if resume else self.initial_state).result()
        write(out / 'model-info.json', self.train.get_info().model_dump())
        assert start_step < steps
        self.publish(f'{name}-{start_step:03d}'.replace('_', '-'))
        if start_step == 0 and (not teacher or domain == DOMAINS[0]):  # Evaluate the shared base once.
            self.evaluate(out, 0)
        for step in range(start_step, steps):
            update = step + 1
            check(self.root)
            begin = time.monotonic()
            offset = cfg['prompt_offset'] + (step - start_step) * per_domain
            work = [(d, self.data[d]['train'][self.order[d][(offset + i) % len(self.order[d])]])
                    for d in domains for i in range(per_domain)]
            self.log('step_start', run=name, step=update)
            groups = self.sample_batch(out, work, step, 'train', cfg['group_size'])
            sample_seconds = time.monotonic() - begin
            records = [r for group in groups for r in group]
            score_start = time.monotonic()
            if teacher:  # Group-normalized rewards on every token.
                for group in groups:
                    scores = [r['score'] for r in group]
                    mean, std = statistics.mean(scores), statistics.stdev(scores) if len(scores) > 1 else 0
                    for r in group:
                        r['advantages'] = [(r['score'] - mean) / (std + 1e-6)] * len(r['tokens'])
            else:
                self.score_with_teachers(out, step, records)
            teacher_seconds = time.monotonic() - score_start
            learn_start = time.monotonic()
            # Keep the documented Fireworks operation order; never retry optimizer mutations.
            backward = self.train.forward_backward([training_datum(r) for r in records],
                                                   loss_fn='ppo' if teacher else 'importance_sampling',
                                                   loss_fn_config=PPO_CLIP if teacher else None).result()
            optimizer = self.train.optim_step(tinker.AdamParams(
                learning_rate=cfg['learning_rate'], **(TEACHER_ADAM if teacher else STUDENT_ADAM))).result()
            deltas = []
            for r, output in zip(records, backward.loss_fn_outputs, strict=True):
                logprobs = response_logprobs(r, output)
                deltas.extend(x - y for x, y in zip(logprobs, r['logprobs']))
                append(out / 'learner-scores.jsonl', {'step': step, 'domain': r['domain'], 'index': r['index'],
                                                      'sample': r['sample'], 'logprobs': logprobs,
                                                      'advantages': r['advantages']})
            train_seconds = time.monotonic() - learn_start
            sampler_path = self.publish(f'{name}-{update:03d}'.replace('_', '-'))
            metric = {'step': update, 'samples': len(records), 'sampling_seconds': sample_seconds,
                      'teacher_seconds': teacher_seconds, 'train_seconds': train_seconds,
                      'step_seconds': time.monotonic() - begin, 'accepted_policy_lag': 0,
                      'prompt_tokens': sum(len(r['prompt_tokens']) for r in records),
                      'response_tokens': sum(len(r['tokens']) for r in records),
                      'scores': {d: statistics.mean(r['score'] for r in records if r['domain'] == d) for d in domains},
                      'nonzero_advantage_samples': sum(any(v != 0 for v in r['advantages']) for r in records),
                      'response_train_sample_logprob_delta_abs_mean': statistics.mean(abs(x) for x in deltas),
                      'forward_backward_metrics': backward.metrics, 'optimizer_metrics': optimizer.metrics}
            append(out / 'metrics.jsonl', metric)
            self.log('step_complete', run=name, **metric)
            if update % cfg['eval_every'] == 0 or update == steps:
                state = self.train.save_state(f'{name}-{update:03d}'.replace('_', '-')).result().path
                checkpoint = {'step': update, 'prompt_offset': offset + per_domain, 'sampler_path': sampler_path,
                              'state_path': state}
                append(out / 'checkpoints.jsonl', checkpoint)
                evaluation = self.evaluate(out, update)
                if teacher:
                    accuracy = evaluation[domain]['accuracy']
                    if best is None or accuracy > best['evaluation'][domain]['accuracy']:
                        best = {'step': update, 'evaluation': evaluation, 'checkpoint': checkpoint,
                                'target': TEACHER_TARGETS[domain], 'stopping_threshold': cfg['stopping_threshold']}
                    if accuracy >= cfg['stopping_threshold']:
                        break
        if teacher:  # Keep the best evaluated checkpoint, even below target.
            self.train.load_state_with_optimizer(best['checkpoint']['state_path']).result()
            self.selected[domain] = {**best, 'model_id': self.train.get_info().model_id}
            write(self.root / 'teacher-selection.json', self.selected)
        self.log('stage_complete', run=name, updates=update, elapsed_seconds_stage=time.monotonic() - start)
        write(out / 'complete.json', {'updates': update, 'elapsed_seconds': time.monotonic() - start})


def prepare(root, args):
    """Record the inputs, verify the account, freeze the source, and re-execute the frozen copy."""
    os.environ['FIREWORKS_API_KEY'] = credentials.api_key('FIREWORKS_API_KEY', args.settings.get('api_key_profile_label'))
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'teacher-config.json', read_json(args.teacher_config) if args.teacher_config else {})
    write(root / 'resume-plan.json', read_json(args.resume_plan) if args.resume_plan else {})
    write(root / 'tokenizer.json', read_json(args.tokenizer_info))
    write(root / 'account.json', verify_account(args.account))
    for package in ['fireworks-ai', 'tinker', 'transformers', 'torch']:
        append(root / 'versions.jsonl', {'package': package, 'version': importlib.metadata.version(package)})
    for part in ['hosted/fireworks', 'shared', 'data']:
        shutil.copytree(REPO / part, root / 'source-repo' / part, ignore=shutil.ignore_patterns('__pycache__'))
    os.execv(sys.executable, [
        sys.executable, str(root / 'source-repo/hosted/fireworks/campaign.py'), '--root', str(root),
        '--tokenizer-info', str(root / 'tokenizer.json'), '--prior-spend-usd', str(args.prior_spend_usd),
        '--budget-usd', str(args.budget_usd), '--rate-usd-hour', str(args.rate_usd_hour), '--prepared'])


def run(root, args):
    info = read_json(root / 'tokenizer.json')
    account = read_json(root / 'account.json')['account']
    rows = {d: {split: puzzles.load_split(d, split) for split in ('train', 'dev')} for d in DOMAINS}
    campaign = Campaign(root, AutoTokenizer.from_pretrained(info['path'], local_files_only=True), rows,
                        read_json(root / 'teacher-config.json'), read_json(root / 'resume-plan.json'))
    campaign.render_prompts()
    write(root / 'data-manifest.json', puzzles.manifest())
    # Resumed teachers must come from a run with the same account, tokenizer and data.
    from fireworks.training.sdk.client import _validate_checkpoint_ref
    for domain, plan in campaign.resume_plan.items():
        assert domain in DOMAINS and 0 <= plan['step'] <= campaign.teacher_settings['max_steps']
        _validate_checkpoint_ref(plan['restore_reference'])
        if plan.get('best'):
            _validate_checkpoint_ref(plan['best']['checkpoint']['state_path'])
        source = Path(plan['source_root'])
        assert read_json(source / 'account.json')['account'] == account
        assert read_json(source / 'tokenizer.json')['revision'] == info['revision']
        assert read_json(source / 'data-manifest.json') == read_json(root / 'data-manifest.json')
    run_id = 'opd-' + datetime.now(timezone.utc).strftime('%m%d%H%M%S')
    resources = {'account': account, 'trainer_id': run_id + '-train', 'deployment_id': run_id + '-sample',
                 'budget_start_unix': time.time(), 'prior_spend_estimate_usd': args.prior_spend_usd,
                 'budget_usd': args.budget_usd, 'stop_estimate_usd': args.budget_usd - CLEANUP_RESERVE_USD,
                 'conservative_usd_hour': args.rate_usd_hour, 'controller_pid': os.getpid()}
    write(root / 'resources.json', resources)
    # The watchdog runs in its own session, so it outlives a killed controller.
    with (root / 'watchdog.log').open('w') as log:
        watchdog = subprocess.Popen(
            [sys.executable, str(root / 'source-repo/hosted/fireworks/lifecycle.py'), str(root), '--pid', str(os.getpid())],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(root / 'watchdog.json', {'pid': watchdog.pid})
    for _ in range(60):
        if (root / 'watchdog-ready.json').exists():
            break
        if watchdog.poll() is not None:
            raise RuntimeError('Independent watchdog failed to start')
        time.sleep(1)
    else:
        raise RuntimeError('Independent watchdog did not become ready')
    secret = api_key()
    os.environ['FIREWORKS_API_KEY'] = secret

    class Redact(logging.Filter):
        def filter(self, record):
            record.msg, record.args = record.getMessage().replace(secret, '[REDACTED]'), ()
            return True

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    for handler in logging.getLogger().handlers:
        handler.addFilter(Redact())
    for noisy in ['httpx', 'httpcore']:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        write(root / 'status.json', {'stage': 'provisioning'})
        TrainerJobManager(api_key=secret).create(TrainerJobConfig(
            base_model=MODEL, lora_rank=LORA_RANK, learning_rate=campaign.teacher_settings['learning_rate'],
            training_shape_ref=SHAPE, trainer_replica_count=1, requested_job_id=resources['trainer_id'],
            inactivity_timeout='1800s', display_name=root.name))
        check(root)
        campaign.service = FiretitanServiceClient.from_firetitan_config(
            api_key=secret, base_model=MODEL, tokenizer_model=info['path'], max_lora_rank=LORA_RANK,
            training_shape_id=SHAPE, trainer_job_id=resources['trainer_id'], deployment_id=resources['deployment_id'],
            max_context_length=CONTEXT_LENGTH, replica_count=1, trainer_replica_count=1,
            hot_load_transition_type='SYNC', disable_speculative_decoding=True, cleanup_trainer_on_close=False,
            cleanup_deployment_on_close=None, trainer_timeout_s=900, trainer_pending_timeout_s=900,
            deployment_timeout_s=900, wait_for_trainer_before_deployment=False)
        for domain in [*DOMAINS, None]:
            write(root / 'status.json', {'stage': 'teacher', 'domain': domain} if domain else {'stage': 'student'})
            campaign.train_stage(domain)
        write(root / 'status.json', {'stage': 'training_complete'})
    except BaseException:
        (root / 'error.txt').write_text(traceback.format_exc())
        write(root / 'status.json', {'stage': 'budget_stopped' if (root / 'STOP').exists() else 'failed'})
        raise
    finally:
        (root / 'TRAINING_FINISHED').write_text(utc_now())
        try:  # The independent watchdog also handles abrupt controller exits.
            cleanup(root, resources)
        except Exception as error:
            event(root, 'controller_cleanup_error', error=str(error))
        for client in (campaign.sampler, campaign.service):
            if client is not None:
                client.close()
        campaign.pool.shutdown(wait=False, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--tokenizer-info', type=Path, required=True)
    parser.add_argument('--account', help='Default: config/fireworks.local.json, then $FIREWORKS_ACCOUNT_ID')
    parser.add_argument('--config', type=Path, help='Account settings (default: config/fireworks.local.json)')
    parser.add_argument('--teacher-config', type=Path)
    parser.add_argument('--resume-plan', type=Path)
    parser.add_argument('--budget-usd', type=float, default=DEFAULT_BUDGET_USD)
    parser.add_argument('--prior-spend-usd', type=float, default=0)
    parser.add_argument('--rate-usd-hour', type=float, default=RATE_USD_HOUR)
    parser.add_argument('--prepared', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.settings = credentials.local_config('fireworks', args.config)
    args.account = args.account or args.settings.get('account') or os.environ.get('FIREWORKS_ACCOUNT_ID')
    if not (args.prepared or args.account):
        parser.error('Set --account, config/fireworks.local.json, or $FIREWORKS_ACCOUNT_ID')
    assert args.budget_usd > CLEANUP_RESERVE_USD and args.rate_usd_hour > 0
    root = args.root.resolve()
    if not args.prepared:
        prepare(root, args)  # Does not return.
    run(root, args)


if __name__ == '__main__':
    main()
