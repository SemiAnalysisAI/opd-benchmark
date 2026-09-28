"""The pieces opd.py and mopd.py share: prompts, rollouts, the distillation update, evaluation, records.

One update (Experiment.update):
  1. The student samples one response per training prompt.
  2. The frozen teacher of that prompt's domain scores every response token (a prefill, no
     generation). Scoring starts as soon as a response finishes, while others still generate.
  3. The per-token advantage is teacher logprob minus student logprob: the negative
     sampled-token reverse KL. There is no task reward.
  4. One importance-sampling policy-gradient step on those advantages.
  5. The new weights are saved and published to the sampler for the next update.

With --policy-lag 1 (the recipe's default, recipe.MAX_POLICY_LAG), the next update's prompts are sampled
and scored on the current weights while this update trains and publishes: each batch is at most one
optimizer step stale, and the importance-sampling loss uses the logprobs of the weights that sampled it.
--policy-lag 0 is the synchronous variant: every response comes from the current weights.
The protocol defaults are shared/recipe.py's. Everything goes to the run directory: config and
sources, every rollout with its tokens and logprobs, per-update metrics and timings,
evaluations, checkpoints, and events. report.py turns a run into tables and plots.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import sys
import threading
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import tinker  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from tinker_cookbook import renderers  # noqa: E402

from backend import BACKENDS  # noqa: E402
from shared import recipe, scoring  # noqa: E402
from shared.puzzles import REPO_DATA, load_split  # noqa: E402

# Models a run can use: HF repo, the pinned tokenizer revision, and the cookbook renderer (thinking on).
# Hosted weights are provider-controlled; the revision pins the tokenizer and chat template.
MODELS = {
    'qwen3.6-35b-a3b': (*recipe.BASE_MODEL, 'qwen3_5'),
    # Qwen3.8's template adds a reasoning-effort system preamble; xhigh is the HF default.
    'qwen3.8-27b': ('Qwen/Qwen3.8-27B', '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0', 'qwen3_8_xhigh_reasoning'),
}
DEFAULT_MODEL = 'qwen3.6-35b-a3b'
MODEL, REVISION, RENDERER = MODELS[DEFAULT_MODEL]  # The recipe's model, used by sft.py and the Fireworks backend.
SEED_STRIDE = 100_000  # Training prompt i of update u samples with seed + u*SEED_STRIDE + i.
ADAM = {'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8, 'weight_decay': 0.0, 'grad_clip_norm': 1.0}


def add_protocol_arguments(parser):
    """The flags opd.py and mopd.py share. Defaults are shared/recipe.py's."""
    parser.add_argument('--output', type=Path, required=True, help='A new run directory')
    parser.add_argument('--backend', choices=sorted(BACKENDS), default='tinker')
    parser.add_argument('--student-model', choices=sorted(MODELS), default=DEFAULT_MODEL)
    parser.add_argument('--teacher-model', choices=sorted(MODELS), default=None,
                        help='The teachers\' base model, for their prompt template (default: the student\'s). They must share '
                             'the student\'s vocabulary: they score the student\'s tokens after a prompt in their own template.')
    parser.add_argument('--updates', type=int, default=recipe.UPDATES)
    parser.add_argument('--prompts', type=int, default=recipe.PROMPTS_PER_UPDATE, help='Training prompts per update')
    parser.add_argument('--learning-rate', type=float, default=10 * recipe.LEARNING_RATE,
                        help="The recipe's full-finetuning LR times the cookbook's 10x for LoRA")
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--max-tokens', type=int, default=recipe.MAX_RESPONSE_TOKENS)
    parser.add_argument('--eval-every', type=int, default=recipe.EVAL_INTERVAL)
    parser.add_argument('--eval-examples', type=int, default=100, help='First N dev problems of every domain')
    parser.add_argument('--eval-samples', type=int, default=3)
    parser.add_argument('--eval-temperature', type=float, default=1.0)
    parser.add_argument('--skip-initial-eval', action='store_true',
                        help='Skip the update-0 evaluation (a protocol change; the base model is measured elsewhere)')
    parser.add_argument('--no-eval', action='store_true', help='Run no evaluations at all (a protocol change)')
    parser.add_argument('--save-every', type=int, help='Optimizer-state checkpoint interval (default: --eval-every); '
                                                       'the last update is always saved')
    parser.add_argument('--policy-lag', type=int, choices=(0, 1), default=recipe.MAX_POLICY_LAG,
                        help='1: sample the next batch while this update trains (async, one step stale); 0: synchronous')
    parser.add_argument('--concurrency', type=int, default=256)
    parser.add_argument('--resume', help='Optimizer-state path from checkpoints.jsonl; needs --start-update')
    parser.add_argument('--start-update', type=int, default=0)


def check_protocol_arguments(parser, args):
    if bool(args.resume) != bool(args.start_update): parser.error('--resume and --start-update go together')
    if not 0 <= args.start_update < args.updates: parser.error('require 0 <= start-update < updates')
    if args.save_every is not None and args.save_every <= 0: parser.error('--save-every must be positive')
    for key in ('updates', 'prompts', 'rank', 'max_tokens', 'eval_every', 'eval_examples', 'eval_samples', 'concurrency'):
        if getattr(args, key) <= 0: parser.error(f'--{key.replace("_", "-")} must be positive')
    if args.eval_examples > min(recipe.EVAL_EXAMPLES.values()): parser.error('--eval-examples exceeds a dev split')


def load_renderer(model):
    """The pinned tokenizer and cookbook renderer of a MODELS entry."""
    repo, revision, renderer = model
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    return tokenizer, renderers.get_renderer(renderer, tokenizer, model_name=repo)


def utc():
    return datetime.now(timezone.utc).isoformat()


def advantages(teacher_logprobs, student_logprobs):
    """Per-token teacher minus student logprob: the negative sampled-token reverse KL."""
    assert len(teacher_logprobs) == len(student_logprobs)
    values = [t - s for t, s in zip(teacher_logprobs, student_logprobs)]
    assert all(math.isfinite(v) for v in values)
    return values


def policy_datum(prompt, tokens, logprobs, advantage):
    """An importance-sampling datum; prompt positions carry zero advantage."""
    assert tokens and len(tokens) == len(logprobs) == len(advantage)
    pad = len(prompt) - 1
    shape = [pad + len(tokens)]
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(prompt + tokens[:-1]), loss_fn_inputs={
        'target_tokens': tinker.TensorData(data=[0] * pad + tokens, dtype='int64', shape=shape),
        'logprobs': tinker.TensorData(data=[0.0] * pad + logprobs, dtype='float32', shape=shape),
        'advantages': tinker.TensorData(data=[0.0] * pad + advantage, dtype='float32', shape=shape)})


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


class Experiment:
    """One run directory: a student distilled from `teachers` ({domain: teacher sampler path})."""

    def __init__(self, args, kind, teachers):
        self.args, self.kind, self.teacher_paths = args, kind, teachers
        self.domains = list(teachers)
        self.out = args.output.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out/'rollouts').mkdir()
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=args.concurrency)
        self.record_inputs()
        self.backend = BACKENDS[args.backend](experiment=f'{kind}-{self.out.name}')
        (self.out/'account.json').write_text(json.dumps(self.backend.account, indent=2))
        self.student_model = MODELS[args.student_model]
        (self.out/'capabilities.json').write_text(json.dumps(self.backend.capabilities(self.student_model[0]), indent=2, default=str))
        self.tokenizer, self.renderer = load_renderer(self.student_model)
        self.stop = self.renderer.get_stop_sequences()
        teacher_model = MODELS[args.teacher_model or args.student_model]
        # A teacher on another base model reads the prompt in its own template, then scores the student's tokens.
        self.teacher_tokenizer, self.teacher_renderer = (
            (self.tokenizer, self.renderer) if teacher_model == self.student_model else load_renderer(teacher_model))
        self.backend.use_tokenizers(self.tokenizer, self.teacher_tokenizer)
        self.data = {d: {s: load_split(d, s) for s in ('train', 'dev')} for d in recipe.DOMAINS}
        self.order = {}  # An independent seeded shuffle per domain: OPD and MOPD see one prompt order.
        for d in self.domains:
            self.order[d] = list(range(len(self.data[d]['train'])))
            random.Random(args.seed + recipe.DOMAINS.index(d)).shuffle(self.order[d])
        self.log('ready', kind=kind, backend=args.backend, account=self.backend.account)

    def record_inputs(self):
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(self.args).items()}
        student = MODELS[self.args.student_model]
        teacher = MODELS[self.args.teacher_model or self.args.student_model]
        config.update(kind=self.kind, domains=self.domains, model=student[0], tokenizer_revision=student[1], renderer=student[2],
                      teacher_model=teacher[0], teacher_tokenizer_revision=teacher[1], teacher_renderer=teacher[2],
                      adam=ADAM, loss='importance_sampling', advantage='teacher_logprob - student_logprob',
                      started_utc=utc())
        (self.out/'config.json').write_text(json.dumps(config, indent=2))
        (self.out/'teachers.json').write_text(json.dumps(self.teacher_paths, indent=2))
        packages = ('tinker', 'tinker-cookbook', 'transformers', 'torch')
        (self.out/'versions.json').write_text(json.dumps({p: importlib.metadata.version(p) for p in packages}, indent=2))
        (self.out/'source').mkdir()
        here = Path(__file__).parent
        for path in (*here.glob('*.py'), REPO/'shared/recipe.py', REPO/'shared/scoring.py'):
            shutil.copyfile(path, self.out/'source'/path.name)
        (self.out/'data-manifest.json').write_text((REPO_DATA/'manifest.json').read_text())

    # Records.

    def append(self, filename, row):
        with self.lock, (self.out/filename).open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')

    def log(self, event, **fields):
        row = {'event': event, 'utc': utc(), 'elapsed_seconds': time.monotonic() - self.started, **fields}
        self.append('events.jsonl', row)
        print(json.dumps(row), flush=True)

    def save_rollouts(self, name, rows):
        with gzip.open(self.out/'rollouts'/f'{name}.jsonl.gz', 'wt') as f:
            f.writelines(json.dumps(r) + '\n' for r in rows)

    # Prompts and sampling.

    def prompt(self, row, teacher=False):
        """The pinned HF chat template with thinking on; it must equal the cookbook renderer's tokens.
        With teacher=True, the teachers' template (the same unless the teachers are another base model)."""
        tokenizer, renderer = (self.teacher_tokenizer, self.teacher_renderer) if teacher else (self.tokenizer, self.renderer)
        ids = tokenizer.apply_chat_template(row['prompt'], tokenize=True, return_dict=False,
                                            add_generation_prompt=True, enable_thinking=recipe.ENABLE_THINKING)
        assert ids == renderer.build_generation_prompt(row['prompt']).to_ints()
        assert len(ids) + self.args.max_tokens <= recipe.CONTEXT_LENGTH
        return ids

    def params(self, temperature, seed=None):
        return tinker.SamplingParams(max_tokens=self.args.max_tokens, temperature=temperature, top_p=1.0, top_k=-1,
                                     stop=self.stop, seed=seed)

    def train_prompts(self, domain, update, count):
        """Update `update`'s `count` prompts of one domain: the next rows of its shuffled training split."""
        order = self.order[domain]
        return [(domain, self.data[domain]['train'][order[(update*count + i) % len(order)]]) for i in range(count)]

    def rollout(self, student, domain, row, seed, t0):
        """Sample one response, then score it with its domain's teacher. Times are seconds after t0."""
        prompt = self.prompt(row)
        begin = time.monotonic()
        sequence = student.sample(tinker.ModelInput.from_ints(prompt), num_samples=1,
                                  sampling_params=self.params(1.0, seed)).result().sequences[0]
        sampled = time.monotonic()
        tokens, logprobs = list(sequence.tokens), list(sequence.logprobs)
        teacher_prompt = self.prompt(row, teacher=True)
        full = self.teachers[domain].compute_logprobs(tinker.ModelInput.from_ints(teacher_prompt + tokens)).result()
        scored = time.monotonic()
        teacher = full[len(teacher_prompt):]  # Entry j is the teacher's logprob of token j of the full sequence.
        response = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return {'domain': domain, 'index': row['_index'], 'seed': seed, 'prompt_tokens': prompt, 'tokens': tokens,
                'logprobs': logprobs, 'teacher_logprobs': teacher, 'advantages': advantages(teacher, logprobs),
                'response': response, 'score': scoring.score(response, row['label']),
                'stop_reason': sequence.stop_reason, 'submitted': begin - t0, 'sampled': sampled - t0,
                'scored': scored - t0}

    # Evaluation: the teacher benchmark protocol (hosted/tinker/sft.py) on every recipe domain,
    # including domains this run does not train on.

    def evaluate(self, student, update):
        a = self.args
        start = time.monotonic()

        def one(domain, row):
            result = student.sample(tinker.ModelInput.from_ints(self.prompt(row)), num_samples=a.eval_samples,
                                    sampling_params=self.params(a.eval_temperature)).result()
            records = []
            for j, sequence in enumerate(result.sequences):
                text = self.tokenizer.decode(sequence.tokens, skip_special_tokens=True)
                records.append({'update': update, 'domain': domain, 'index': row['_index'], 'sample': j,
                                'tokens': list(sequence.tokens), 'response': text,
                                'score': scoring.score(text, row['label']), 'stop_reason': sequence.stop_reason})
            return records

        futures = [self.pool.submit(one, d, row) for d in recipe.DOMAINS for row in self.data[d]['dev'][:a.eval_examples]]
        records = [r for f in futures for r in f.result()]
        self.save_rollouts(f'eval-{update:03d}', records)
        result = {'update': update, 'seconds': time.monotonic() - start}
        for d in recipe.DOMAINS:
            rows = [r for r in records if r['domain'] == d]
            by_problem = {}
            for r in rows:
                by_problem.setdefault(r['index'], []).append(r['score'])
            result[d] = {'problems': len(by_problem), 'samples': len(rows), 'accuracy': statistics.mean(r['score'] for r in rows),
                         f'pass_at_{a.eval_samples}': statistics.mean(max(v) for v in by_problem.values()),
                         'mean_tokens': statistics.mean(len(r['tokens']) for r in rows),
                         'truncated': sum(r['stop_reason'] == 'length' for r in rows)}
        self.append('evaluations.jsonl', result)
        self.log('evaluation', **result)

    # Training.

    def start(self):
        """The training client, the frozen teachers, and a sampler of the starting weights."""
        a = self.args
        self.teachers = {d: self.backend.sampler(path=p) for d, p in self.teacher_paths.items()}
        if a.resume:
            train = self.backend.resume_student(a.resume)
        else:
            train = self.backend.new_student(self.student_model[0], a.rank, a.seed, {'run': self.out.name, 'kind': self.kind})
        try:
            (self.out/'model-info.json').write_text(train.get_info().model_dump_json(indent=2))
        except Exception as error:  # Not every service implements get_info.
            (self.out/'model-info.json').write_text(json.dumps({'unavailable': str(error)[:300]}))
        student, _ = self.backend.sync(train, f'update-{a.start_update:03d}')
        return train, student

    def submit(self, student, update, work, version):
        """Start sampling and scoring update `update`'s `work`, a list of (domain, row) prompts, on `student`:
        the weights after `version` optimizer steps. Rollout times are seconds after this submission."""
        t0 = time.monotonic()
        self.log('rollouts_submitted', update=update + 1, prompts=len(work), policy_version=version)
        futures = [self.pool.submit(self.rollout, student, d, row, self.args.seed + update*SEED_STRIDE + i, t0)
                   for i, (d, row) in enumerate(work)]
        return {'update': update, 'version': version, 'futures': futures}

    def update(self, train, batch, prefetch):
        """One distillation update on a submitted batch. Returns the new student sampler and the next batch:
        prefetch() submits it on the weights that sampled this one once this batch is in (async), or returns
        None (synchronous, or the last update) for the caller to submit on the new sampler."""
        a = self.args
        update = batch['update']
        t0 = time.monotonic()
        self.log('update_start', update=update + 1, prompts=len(batch['futures']))
        rollouts = [f.result() for f in batch['futures']]
        rollout_seconds = time.monotonic() - t0  # The exposed wait; with lookahead, sampling began during the last update.
        lag = update - batch['version']
        assert 0 <= lag <= a.policy_lag, 'Refuse a stale or future-policy batch'
        upcoming = prefetch()
        self.save_rollouts(f'update-{update+1:03d}', [{'update': update + 1, 'policy_lag': lag, **r} for r in rollouts])

        train_start = time.monotonic()
        datums = [policy_datum(r['prompt_tokens'], r['tokens'], r['logprobs'], r['advantages']) for r in rollouts]
        fb = train.forward_backward(datums, loss_fn='importance_sampling')
        opt = train.optim_step(tinker.AdamParams(learning_rate=a.learning_rate, **ADAM))
        fb, opt = fb.result(), opt.result()
        train_seconds = time.monotonic() - train_start
        # Trainer-versus-sampler logprob mismatch on the sampled tokens, before this update's step, when the
        # provider returns per-token logprobs from forward_backward (Tinker does; Fireworks returns a loss).
        mismatch = [float(x) - y for r, o in zip(rollouts, fb.loss_fn_outputs, strict=True) if 'logprobs' in o
                    for x, y in zip(o['logprobs'].data[len(r['prompt_tokens']) - 1:], r['logprobs'], strict=True)]

        sync_start = time.monotonic()
        student, sampler_path = self.backend.sync(train, f'update-{update+1:03d}')
        sync_seconds = time.monotonic() - sync_start
        self.append('metrics.jsonl', self.update_metrics(update, rollouts, t0, rollout_seconds, train_seconds,
                                                         sync_seconds, mismatch, sampler_path, fb, opt, lag))
        return student, upcoming

    def update_metrics(self, update, rollouts, t0, rollout_seconds, train_seconds, sync_seconds, mismatch,
                       sampler_path, fb, opt, lag):
        sampled, scored = [r['sampled'] for r in rollouts], [r['scored'] for r in rollouts]
        response_tokens = sum(len(r['tokens']) for r in rollouts)
        prompt_tokens = sum(len(r['prompt_tokens']) for r in rollouts)
        by_domain = {}
        for d in self.domains:
            rs = [r for r in rollouts if r['domain'] == d]
            by_domain[d] = {'responses': len(rs), 'score': statistics.mean(r['score'] for r in rs),
                            'reverse_kl': -statistics.mean(v for r in rs for v in r['advantages']),
                            'mean_response_tokens': statistics.mean(len(r['tokens']) for r in rs)}
        update_seconds = time.monotonic() - t0
        metric = {
            'update': update + 1, 'policy_lag': lag, 'responses': len(rollouts), 'learning_rate': self.args.learning_rate,
            'prompt_tokens': prompt_tokens, 'response_tokens': response_tokens,
            'max_response_tokens': max(len(r['tokens']) for r in rollouts),
            'truncated': sum(r['stop_reason'] == 'length' for r in rollouts), 'by_domain': by_domain,
            'timing': {  # Seconds. Rollout is the wait for sampling plus the overlapped teacher scoring; response
                         # times are from the batch's submission, which is earlier than this update with lookahead.
                'rollout': rollout_seconds, 'train': train_seconds, 'sync': sync_seconds, 'update': update_seconds,
                'first_response': min(sampled), 'median_response': statistics.median(sampled),
                'p90_response': percentile(sampled, 0.9), 'last_response': max(sampled),
                'teacher_tail': max(scored) - max(sampled),
                'teacher_request_median': statistics.median(r['scored'] - r['sampled'] for r in rollouts)},
            'throughput': {  # Tokens per second of wall time.
                'sampled_tokens_per_second': response_tokens / max(sampled),
                'trained_tokens_per_second': (prompt_tokens + response_tokens) / train_seconds,
                'update_response_tokens_per_second': response_tokens / update_seconds},
            'train_sampler_logprob_delta_mean': statistics.mean(mismatch) if mismatch else None,
            'train_sampler_logprob_delta_abs_mean': statistics.mean(abs(x) for x in mismatch) if mismatch else None,
            'sampler_path': sampler_path, 'forward_backward_metrics': fb.metrics, 'optimizer_metrics': opt.metrics}
        self.log('update_complete', **{k: v for k, v in metric.items() if k != 'forward_backward_metrics'})
        return metric

    def checkpoint(self, train, update):
        state = train.save_state(name=f'state-{update:03d}').result().path
        self.append('checkpoints.jsonl', {'update': update, 'state_path': state})
        self.log('checkpoint', update=update, state_path=state)

    def train_loop(self, prompts_for_update):
        """Evaluate, then run every update on prompts_for_update(update), with scheduled checkpoints and evaluations."""
        a = self.args
        save_every = a.save_every or a.eval_every
        train, student = self.start()
        if a.start_update == 0 and not (a.skip_initial_eval or a.no_eval):
            self.evaluate(student, 0)
        batch = None
        for update in range(a.start_update, a.updates):
            batch = batch or self.submit(student, update, prompts_for_update(update), version=update)

            def prefetch():  # Async: the next batch samples on the weights that sampled this one while it trains.
                if a.policy_lag and update + 1 < a.updates:
                    return self.submit(student, update + 1, prompts_for_update(update + 1), version=update)

            student, batch = self.update(train, batch, prefetch)
            done = update + 1
            if done % save_every == 0 or done == a.updates:
                self.checkpoint(train, done)
            if not a.no_eval and (done % a.eval_every == 0 or done == a.updates):
                self.evaluate(student, done)
        self.log('complete', updates=a.updates - a.start_update)


def run(experiment_factory, prompts_for_update):
    """Run an experiment, recording any failure in its directory."""
    experiment = None
    try:
        experiment = experiment_factory()
        experiment.train_loop(lambda update: prompts_for_update(experiment, update))
    except BaseException as error:
        if experiment:
            experiment.log('failed', error_type=type(error).__name__, message=str(error))
            (experiment.out/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        if experiment:
            experiment.pool.shutdown(wait=True, cancel_futures=True)
            if hasattr(experiment, 'backend'):
                experiment.backend.close()
