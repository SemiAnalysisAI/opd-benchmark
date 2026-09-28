"""SFT teachers from teacher traces, and the teacher benchmark, on a Tinker-compatible API. Thinking on.

`train` fits one LoRA teacher for one domain on that domain's traces, then benchmarks it.
`bench` benchmarks the base model, or any sampler checkpoint, the same way: the first
`--bench-examples` problems of each dev split, `--bench-samples` responses each.

The defaults reproduce the released teachers: every verified trace of
semianalysisai/Qwen3.6-35B-A3B-teacher-sft-caesar-geometry (sampled from the frozen
recipe.TEACHERS), one epoch, rank 64, and a 100-problem x 3-sample benchmark at temperature 1.
Every setting is a flag and is recorded in the run's config.json, so other teachers
(fewer or easier traces, another LR, rank, or epoch count, or another trace dataset
with the same layout) are one command away:

    python sft.py train --domain caesar_cipher --output /abs/teacher-caesar-cipher
    python sft.py train --domain caesar_cipher --output /abs/t --min-correct 3 --traces-per-prompt 1
    python sft.py bench --output /abs/baseline [--checkpoint tinker://.../sampler_weights/final]
    python sft.py train --domain caesar_cipher --output /abs/t --backend fireworks   # see backend.py

A trace dataset needs data/<domain>/train-*.parquet files with columns id, task, question and
messages (system, user, assistant; the assistant turn is `<think>\n...</think>...`), and the
system prompt and scorer labels of this repository's data/. The trace filters also need
correct_of_3, source_index and sample_index. Adapted from the cookbook's recipes/sl_loop.py.
No model weights are downloaded.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import sys
import threading
import time
import traceback

import tinker
from transformers import AutoTokenizer
from tinker_cookbook import renderers
from tinker_cookbook.hyperparam_utils import get_lr

from common import MODEL, RENDERER, REVISION  # common.py puts the repository on sys.path.
from backend import BACKENDS
from shared import recipe, scoring
from shared.puzzles import REPO_DATA, load_split

DOMAINS = recipe.DOMAINS
# Defaults of every setting; each is also a command-line flag (underscores become dashes).
DEFAULTS = {
    # Traces: the released dataset, pinned. By default every trace is used; the filters keep prompts the
    # teacher solved at least `min_correct` of 3 times, at most `traces_per_prompt` traces each (the lowest
    # sample indices; 0 keeps all), and then at most `max_rows` seeded-random rows (0 keeps all).
    'dataset': 'semianalysisai/Qwen3.6-35B-A3B-teacher-sft-caesar-geometry',
    'revision': 'b21bb3e75b1ca903f4b69bd609adc89bd99543f1',
    'min_correct': 1, 'traces_per_prompt': 0, 'max_rows': 0,
    # Training, cookbook SFT defaults: sum-over-token cross entropy on the completion, the cookbook's
    # recommended LoRA LR for this model (about 5e-4), linear decay to zero. Each epoch is a fresh seeded
    # shuffle whose final partial batch is dropped, so every update sees the same number of sequences.
    'learning_rate': get_lr(MODEL), 'lr_schedule': 'linear', 'batch_size': 128, 'epochs': 1, 'rank': 64,
    'save_every': 25,
    # The benchmark: the first 100 problems of each dev split, 3 samples each at temperature 1, the
    # sampling used to generate the dataset and to report the GRPO teachers' held-out avg@8.
    'bench_examples': 100, 'bench_samples': 3, 'bench_temperature': 1.0,
}
ADAM = {'beta1': 0.9, 'beta2': 0.95, 'eps': 1e-8, 'weight_decay': 0.0, 'grad_clip_norm': 0.0}


def lr_at(step, steps, peak, schedule='linear'):
    """The learning rate of update `step` (0-based) of `steps`: linear or cosine decay to zero, or constant."""
    if schedule == 'constant':
        return peak
    if schedule == 'cosine':
        return peak * 0.5 * (1 + math.cos(math.pi * step / steps))
    assert schedule == 'linear'
    return peak * (1 - step / steps)


def select_traces(rows, min_correct=1, traces_per_prompt=0, max_rows=0, seed=0):
    """Apply the trace filters, keeping the dataset's row order."""
    if min_correct > 1:
        rows = [r for r in rows if r['correct_of_3'] >= min_correct]
    if traces_per_prompt:
        kept = {}
        for r in sorted(rows, key=lambda r: (r['source_index'], r['sample_index'])):
            kept.setdefault(r['source_index'], [])
            if len(kept[r['source_index']]) < traces_per_prompt:
                kept[r['source_index']].append(r['id'])
        keep = {i for ids in kept.values() for i in ids}
        rows = [r for r in rows if r['id'] in keep]
    if max_rows and len(rows) > max_rows:
        keep = set(random.Random(seed).sample(range(len(rows)), max_rows))
        rows = [r for i, r in enumerate(rows) if i in keep]
    return rows


def epoch_order(n, epochs, seed):
    """Row indices for every update: one seeded shuffle per epoch, the final partial batch dropped later."""
    orders = []
    for epoch in range(epochs):
        order = list(range(n))
        random.Random(seed + 1000 * epoch).shuffle(order)
        orders.append(order)
    return orders


def sft_datum(prompt_tokens, completion_tokens):
    """Next-token datum trained on the completion only (thinking, answer, and end-of-turn)."""
    assert prompt_tokens and completion_tokens
    sequence = prompt_tokens + completion_tokens
    weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens)
    shape = [len(sequence) - 1]
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(sequence[:-1]), loss_fn_inputs={
        'target_tokens': tinker.TensorData(data=sequence[1:], dtype='int64', shape=shape),
        'weights': tinker.TensorData(data=weights, dtype='float32', shape=shape)})


def sequence_nll(output, prompt_length, completion_length):
    """One sequence's summed completion NLL from a cross_entropy forward_backward output.

    Tinker returns per-token logprobs; Fireworks returns the weighted sum directly as `loss`.
    """
    if 'logprobs' in output:
        logprobs = output['logprobs'].data[prompt_length - 1:]
        assert len(logprobs) == completion_length
        return -sum(logprobs)
    return float(output['loss'].data[0])


def summarize(records, domain, k):
    """avg@k, pass@k, and length statistics of one domain's benchmark records."""
    rows = [r for r in records if r['domain'] == domain]
    problems = {}
    for r in rows:
        problems.setdefault(r['index'], []).append(r['score'])
    return {'problems': len(problems), 'samples': len(rows), 'correct': sum(r['score'] for r in rows),
            'accuracy': statistics.mean(r['score'] for r in rows),
            f'pass_at_{k}': statistics.mean(max(s) for s in problems.values()),
            f'all_{k}_correct': statistics.mean(min(s) for s in problems.values()),
            'mean_tokens': statistics.mean(len(r['tokens']) for r in rows),
            'max_tokens': max(len(r['tokens']) for r in rows),
            'truncated': sum(r['stop_reason'] == 'length' for r in rows),
            'unfinished_thinking': sum('</think>' not in r['response'] for r in rows)}


class Run:
    def __init__(self, args):
        self.args = args
        self.out = args.output.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.write_lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=args.concurrency)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update(adam=ADAM, model=MODEL, tokenizer_revision=REVISION,
                      changed_from_defaults=sorted(k for k, v in DEFAULTS.items() if getattr(args, k) != v))
        self.event('start', config=config)
        (self.out/'config.json').write_text(json.dumps(config, indent=2))
        packages = ('tinker', 'tinker-cookbook', 'transformers', 'torch', 'numpy', 'pyarrow', 'fireworks-ai')
        versions = {}
        for name in packages:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        (self.out/'versions.json').write_text(json.dumps(versions, indent=2))
        (self.out/'source.py').write_text(Path(__file__).read_text())
        (self.out/'backend.py').write_text(Path(__file__).with_name('backend.py').read_text())
        (self.out/'scoring.py').write_text(Path(scoring.__file__).read_text())
        (self.out/'data-manifest.json').write_text((REPO_DATA/'manifest.json').read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.renderer = renderers.get_renderer(RENDERER, self.tokenizer, model_name=MODEL)
        self.stop = self.renderer.get_stop_sequences()
        self.backend = BACKENDS[args.backend](experiment=f'sft-{args.mode}-{self.out.name}')
        (self.out/'account.json').write_text(json.dumps(self.backend.account, indent=2))
        (self.out/'capabilities.json').write_text(json.dumps(self.backend.capabilities(MODEL), indent=2, default=str))
        self.dev = {d: load_split(d, 'dev') for d in DOMAINS}
        self.event('ready', stop=self.stop, tokenizer_revision=REVISION)

    def write(self, filename, row):
        with self.write_lock, (self.out/filename).open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')

    def event(self, name, **data):
        row = {'event': name, 'utc': datetime.now(timezone.utc).isoformat(),
               'elapsed_seconds': time.monotonic() - self.started, **data}
        self.write('events.jsonl', row)
        print(json.dumps(row), flush=True)

    def prompt(self, messages):
        # The pinned HF template, which the traces were generated and verified with; fail if the cookbook differs.
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                                 add_generation_prompt=True, enable_thinking=recipe.ENABLE_THINKING)
        assert ids == self.renderer.build_generation_prompt(messages).to_ints(), 'Pinned HF and cookbook prompt tokens differ'
        return ids

    # Training data.

    def load_traces(self, domain):
        """The domain's selected traces, checked before any training is paid for."""
        import pyarrow.parquet as pq
        from huggingface_hub import snapshot_download
        a = self.args
        root = Path(snapshot_download(a.dataset, repo_type='dataset', revision=a.revision))
        files = sorted((root/'data'/domain).glob('train-*.parquet'))
        assert files, f'No parquet files for {domain}'
        hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        columns = ['id', 'task', 'messages', 'question']
        if a.min_correct > 1 or a.traces_per_prompt:
            columns += ['correct_of_3', 'source_index', 'sample_index']
        released = [r for f in files for r in pq.read_table(f, columns=columns).to_pylist()]
        rows = select_traces(released, a.min_correct, a.traces_per_prompt, a.max_rows, a.seed)
        assert rows, 'The trace filters kept no rows'

        system = self.dev[domain][0]['prompt'][0]['content']
        dev_questions = {r['prompt'][-1]['content'] for d in DOMAINS for r in self.dev[d]}
        for r in rows:
            m = r['messages']
            assert r['task'] == domain and [x['role'] for x in m] == ['system', 'user', 'assistant'], r['id']
            assert m[0]['content'] == system and m[1]['content'] == r['question'], r['id']
            assert r['question'] not in dev_questions, f'Dev problem in SFT data: {r["id"]}'
            text = m[2]['content']
            assert text.startswith('<think>\n') and text.count('</think>') == 1 and text.count('<think>') == 1, r['id']
        record = {'domain': domain, 'repo': a.dataset, 'revision': a.revision, 'files': hashes,
                  'released_rows': len(released), 'rows': len(rows), 'unique_prompts': len({r['question'] for r in rows}),
                  'filters': {k: getattr(a, k) for k in ('min_correct', 'traces_per_prompt', 'max_rows')}}
        (self.out/'sft-data.json').write_text(json.dumps(record, indent=2))
        self.event('traces_loaded', **record)
        return rows

    def tokenize(self, row):
        """Prompt and completion tokens; the completion runs through the end-of-turn token."""
        m = row['messages']
        prompt = self.prompt(m[:2])
        full = self.tokenizer.apply_chat_template(m, tokenize=True, return_dict=False,
                                                  enable_thinking=recipe.ENABLE_THINKING)
        end = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        assert full[:len(prompt)] == prompt, f'Prompt is not a prefix of the rendered trace: {row["id"]}'
        assert full[-2:] == [end, *self.tokenizer.encode('\n')], row['id']
        completion = full[len(prompt):-1]  # Drop the template's newline after <|im_end|>.
        assert len(prompt) + len(completion) <= recipe.CONTEXT_LENGTH, row['id']
        return prompt, completion

    def build_batch(self, rows, orders, step):
        """One update's rows, tokens and datums, and how long rendering them took (seconds)."""
        start = time.monotonic()
        size = self.args.batch_size
        epoch, b = divmod(step, len(rows) // size)
        batch = [rows[i] for i in orders[epoch][b*size:(b+1)*size]]
        tokens = [self.tokenize(r) for r in batch]
        return batch, tokens, [sft_datum(p, c) for p, c in tokens], time.monotonic() - start

    def train(self):
        a = self.args
        rows = self.load_traces(a.domain)
        orders = epoch_order(len(rows), a.epochs, a.seed + DOMAINS.index(a.domain))
        per_epoch = len(rows) // a.batch_size
        steps = per_epoch * a.epochs
        assert per_epoch > 0, f'Fewer rows ({len(rows)}) than one batch ({a.batch_size})'
        assert 0 <= a.start_step < steps
        for i in orders[0][:a.batch_size]:  # Render one batch up front: fail before creating a client.
            self.tokenize(rows[i])
        self.event('plan', rows=len(rows), steps=steps, dropped_per_epoch=len(rows) - per_epoch*a.batch_size,
                   learning_rate=a.learning_rate, lr_schedule=a.lr_schedule)
        if a.checkpoint:
            train = self.backend.resume_student(a.checkpoint)
        else:
            train = self.backend.new_student(MODEL, a.rank, a.seed, {'domain': a.domain, 'run': self.out.name, 'method': 'sft'})
        (self.out/'model-info.json').write_text(train.get_info().model_dump_json(indent=2))
        self.event('training_client_created')
        prefetch = ThreadPoolExecutor(max_workers=1)
        try:
            # One update stays in flight while the next batch is rendered and submitted.
            next_batch = prefetch.submit(self.build_batch, rows, orders, a.start_step)
            pending = None
            self.last_finish = time.monotonic()
            for step in range(a.start_step, steps):
                wait = time.monotonic()
                batch, tokens, datums, build_seconds = next_batch.result()
                timing = {'build_seconds': build_seconds, 'batch_wait_seconds': time.monotonic() - wait}
                if step + 1 < steps:
                    next_batch = prefetch.submit(self.build_batch, rows, orders, step + 1)
                lr = lr_at(step, steps, a.learning_rate, a.lr_schedule)
                timing['submitted'] = time.monotonic()
                fb = train.forward_backward(datums, loss_fn='cross_entropy')
                opt = train.optim_step(tinker.AdamParams(learning_rate=lr, **ADAM))
                timing['submit_seconds'] = time.monotonic() - timing['submitted']
                if pending:
                    self.finish(train, *pending, steps)
                pending = (step, batch, tokens, fb, opt, lr, timing)
            self.finish(train, *pending, steps)
        finally:
            prefetch.shutdown(wait=True, cancel_futures=True)
        sync_start = time.monotonic()
        client, sampler = self.backend.sync(train, 'final')
        self.event('sampler_saved', path=sampler, sync_seconds=time.monotonic() - sync_start)
        results = self.benchmark(client, (a.domain,), sampler)
        final = [json.loads(line) for line in (self.out/'checkpoints.jsonl').read_text().splitlines()][-1]
        assert final['step'] == steps
        teacher = {'domain': a.domain, 'method': 'sft', 'steps': steps, 'sampler_path': sampler,
                   'state_path': final['state_path'], 'benchmark': results[a.domain]}
        with (self.out/'teacher.json').open('x') as f:
            json.dump(teacher, f, indent=2)
        self.event('complete', optimizer_updates=steps - a.start_step, total_optimizer_updates=steps)

    def finish(self, train, step, batch, tokens, fb, opt, lr, timing, steps):
        """Wait for one update, record it, and save optimizer state when scheduled.

        Updates are pipelined (the next is submitted before this one is awaited), so step_seconds
        (submit to result) overlaps its neighbours; interval_seconds, the time between consecutive
        completions, is the throughput measure. It includes any checkpoint save before it, during
        which the trainer keeps working on the next update, so the run-level rate (total tokens over
        total time) is the figure to report.
        """
        fb, opt = fb.result(), opt.result()
        done = time.monotonic()
        interval, self.last_finish = done - self.last_finish, done
        nll, weight = 0.0, 0
        for (prompt, completion), output in zip(tokens, fb.loss_fn_outputs, strict=True):
            nll += sequence_nll(output, len(prompt), len(completion))
            weight += len(completion)
        prompt_tokens = sum(len(p) for p, _ in tokens)
        row = {'step': step + 1, 'utc': datetime.now(timezone.utc).isoformat(), 'sequences': len(batch),
               'learning_rate': lr, 'prompt_tokens': prompt_tokens, 'completion_tokens': weight,
               'train_nll': nll / weight, 'step_seconds': done - timing['submitted'], 'interval_seconds': interval,
               'tokens_per_second': (prompt_tokens + weight) / interval,
               'build_seconds': timing['build_seconds'], 'batch_wait_seconds': timing['batch_wait_seconds'],
               'submit_seconds': timing['submit_seconds'], 'ids': [r['id'] for r in batch],
               'forward_backward_metrics': fb.metrics, 'optimizer_metrics': opt.metrics}
        self.write('metrics.jsonl', row)
        self.event('step_complete', **{k: v for k, v in row.items() if k != 'ids'})
        if (step + 1) % self.args.save_every == 0 or step + 1 == steps:
            save_start = time.monotonic()
            state = train.save_state(name=f'step-{step+1:04d}').result().path
            save_seconds = time.monotonic() - save_start
            self.write('checkpoints.jsonl', {'step': step + 1, 'state_path': state, 'save_seconds': save_seconds})
            self.event('checkpoint', step=step + 1, state_path=state, save_seconds=save_seconds)

    # Benchmark.

    def sample_one(self, client, domain, row):
        tokens = self.prompt(row['prompt'])
        assert len(tokens) + self.args.max_tokens <= recipe.CONTEXT_LENGTH
        start = time.monotonic()
        # Multi-response requests leave the seed unset, as in the cookbook.
        a = self.args
        result = client.sample(tinker.ModelInput.from_ints(tokens), num_samples=a.bench_samples,
            sampling_params=tinker.SamplingParams(max_tokens=a.max_tokens, temperature=a.bench_temperature,
                top_p=1.0, top_k=-1, stop=self.stop)).result()
        assert len(result.sequences) == a.bench_samples
        records = []
        for j, seq in enumerate(result.sequences):
            response = self.tokenizer.decode(seq.tokens, skip_special_tokens=True)
            records.append({'domain': domain, 'index': row['_index'], 'sample': j, 'prompt_tokens': tokens,
                'tokens': seq.tokens, 'response': response, 'score': scoring.score(response, row['label']),
                'stop_reason': seq.stop_reason, 'request_seconds': time.monotonic() - start})
        return records

    def benchmark(self, client, domains, model):
        a = self.args
        start = time.monotonic()
        protocol = {'examples': a.bench_examples, 'samples': a.bench_samples, 'temperature': a.bench_temperature}
        self.event('benchmark_start', domains=domains, **protocol)
        futures = [self.pool.submit(self.sample_one, client, d, row)
                   for d in domains for row in self.dev[d][:a.bench_examples]]
        records = []
        for f in as_completed(futures):
            for r in f.result():
                records.append(r)
                self.write('benchmark-samples.jsonl', r)
        results = {d: summarize(records, d, a.bench_samples) for d in domains}
        seconds = time.monotonic() - start
        latency = sorted(r['request_seconds'] for r in records if r['sample'] == 0)
        sampled = sum(len(r['tokens']) for r in records)
        timing = {'seconds': seconds, 'requests': len(latency), 'sampled_tokens': sampled,
                  'sampled_tokens_per_second': sampled / seconds,
                  'request_seconds': {'median': statistics.median(latency), 'p90': latency[int(0.9 * (len(latency) - 1))],
                                      'max': latency[-1]}}
        with (self.out/'benchmark.json').open('x') as f:
            json.dump({'model': model, 'protocol': protocol, 'max_tokens': a.max_tokens, **timing, **results}, f, indent=2)
        self.event('benchmark', results=results, timing=timing)
        return results

    def execute(self):
        if self.args.mode == 'train':
            self.train()
        else:
            if self.args.checkpoint:
                client = self.backend.sampler(path=self.args.checkpoint)
            else:
                client = self.backend.sampler(model=MODEL)
            self.benchmark(client, self.args.domains, self.args.checkpoint or MODEL)
            self.event('complete')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('train', 'bench'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=sorted(BACKENDS), default='tinker')
    p.add_argument('--domain', choices=DOMAINS, help='train: the teacher domain')
    p.add_argument('--domains', choices=DOMAINS, nargs='+', default=list(DOMAINS), help='bench: domains to benchmark')
    p.add_argument('--checkpoint', help='train: optimizer-state path to resume; bench: sampler path (default: base)')
    p.add_argument('--start-step', type=int, default=0, help='train: completed updates in --checkpoint')
    p.add_argument('--seed', type=int, default=20260921)
    p.add_argument('--max-tokens', type=int, default=recipe.MAX_RESPONSE_TOKENS)
    p.add_argument('--concurrency', type=int, default=DEFAULTS['bench_examples'] * len(DOMAINS))
    for key, value in DEFAULTS.items():
        kind = type(value)
        choices = ('linear', 'cosine', 'constant') if key == 'lr_schedule' else None
        p.add_argument('--' + key.replace('_', '-'), type=kind, default=value, choices=choices,
                       help=f'default: {value}')
    args = p.parse_args()
    for key in ('batch_size', 'epochs', 'rank', 'save_every', 'bench_examples', 'bench_samples'):
        if getattr(args, key) <= 0: p.error(f'--{key.replace("_", "-")} must be positive')
    if not 1 <= args.min_correct <= 3: p.error('--min-correct is between 1 and 3')
    if args.traces_per_prompt < 0 or args.max_rows < 0: p.error('--traces-per-prompt and --max-rows are >= 0')
    if not (math.isfinite(args.learning_rate) and args.learning_rate > 0): p.error('--learning-rate must be positive')
    if args.bench_temperature < 0: p.error('--bench-temperature must be >= 0')
    if args.bench_examples > min(recipe.EVAL_EXAMPLES.values()): p.error('--bench-examples exceeds a dev split')
    if args.mode == 'train' and not args.domain: p.error('train requires --domain')
    if args.start_step and not (args.mode == 'train' and args.checkpoint):
        p.error('--start-step requires a training --checkpoint')
    if args.concurrency <= 0 or args.max_tokens <= 0: p.error('concurrency and max-tokens must be positive')
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
        if run:
            run.pool.shutdown(wait=True, cancel_futures=True)
            if hasattr(run, 'backend'):
                run.backend.close()


if __name__ == '__main__':
    main()
