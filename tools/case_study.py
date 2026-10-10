"""Case-study columns for one Miles or verl run, with the same definitions for both frameworks.

    <python with transformers> tools/case_study.py RESULT_DIR OUT.json

One row per update. Each update's window runs from the previous update's end (or the run start) to
its own end: for Miles, the trainer's per-update `perf` log line; for verl, the modification time of
`rollouts/<N>.jsonl`, which verl writes at the end of update N.

Columns (all per update):
  step_s, train_s, wait_s       the framework's own timers (verl logs update N at step N + 1 and
                                never logs its last update, so those are null for verl's update 20)
  train_tokens                  prompt + response tokens of the trained batch
  train_tok_s_gpu, mfu          train_tokens / train_s / 8 trainer GPUs; MFU at 6 x ACTIVE_PARAMS FLOPs
                                per token against the H100 bf16 peak (attention scores not counted)
  prompt_p50, response_p50      medians over the trained samples (verl: its dump, re-tokenized)
  rollout_e2e_p50_s, rollout_requests, rollout_tok_s_gpu
                                from the policy servers' Prometheus metrics (SGLang or vLLM):
                                end-to-end request latency, finished requests, generated tokens per
                                second per policy GPU. Requests resumed after a weight sync count again.
  filtered, filtered_ratio      finished samples dropped as stale, and dropped / (dropped + 128)
  effective_tok_s_gpu_policy    train_tokens / window / policy GPUs
  teacher_p50_s, teacher_tok_s_gpu, teacher_mfu
                                from the teacher servers' metrics: request latency, prompt (prefill)
                                tokens per second per teacher GPU, MFU at 2 x ACTIVE_PARAMS
  cost_usd                      window x all GPUs of the allocation x H100_USD_PER_HOUR
  train_score                   {task: mean verifier score of the trained batch}
"""
import datetime
import gzip
import json
from pathlib import Path
import re
import statistics
import sys

ACTIVE_PARAMS = 3.0e9  # Qwen3.6-35B-A3B: about 3B parameters active per token (8 of 256 experts)
H100_PEAK_FLOPS = 989.4e12  # dense bf16
H100_USD_PER_HOUR = 6.16
GPUS_PER_NODE, TRAINER_GPUS, GPUS_PER_ENGINE, BATCH = 8, 8, 2, 128
TASKS = ('caesar_cipher', 'simple_geometry')


def campaign(result):
    root = result.parents[1]
    experiment = json.loads((root / 'experiment.json').read_text()) if (root / 'experiment.json').exists() else {}
    framework = json.loads((root / 'package-provenance.json').read_text())['framework']
    domains = experiment.get('domains', list(TASKS))
    layout = experiment.get('layout', 'b200x2')
    nodes = len(json.loads((result / 'start.json').read_text())['nodes'])
    policy_gpus = 12 if layout == 'h100x3' else 6
    return root, framework, domains, nodes, policy_gpus, GPUS_PER_ENGINE * len(domains)


# --- Prometheus captures (both frameworks) ------------------------------------------------------------

def metric_series(result, domains):
    """[(time, {'policy'|'teacher': {metric: {le or '': value}}})], counters summed over servers."""
    rows = []
    for path in sorted(result.glob('*-capture.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            try:  # A capture still being written ends mid-stream.
                for line in stream:
                    row = json.loads(line)
                    rows.append((row['time_unix'], row['prometheus']))
            except (EOFError, json.JSONDecodeError):
                pass
    rows.sort(key=lambda r: r[0])
    wanted = re.compile(r'^(vllm|sglang):(e2e_request_latency_seconds_bucket|prompt_tokens_total|generation_tokens_total)\b')
    latest, out = {}, []
    for t, servers in rows:
        for target, text in servers.items():
            if not isinstance(text, str):
                continue
            for line in text.splitlines():
                if not (m := wanted.match(line)):
                    continue
                series, value = line.rsplit(' ', 1)
                model = re.search(r'model_name="([^"]+)"', series)
                role = 'teacher' if model and model[1].rstrip('/').rsplit('/', 1)[-1] in domains else 'policy'
                le = b[1] if (b := re.search(r'le="([^"]+)"', series)) else ''
                latest[(target, series)] = (role, m[2], le, float(value))
        total = {}
        for role, metric, le, value in latest.values():
            bucket = total.setdefault(role, {}).setdefault(metric, {})
            bucket[le] = bucket.get(le, 0) + value
        out.append((t, total))
    return out


def at(series, t):
    best = {}
    for time, value in series:
        if time > t:
            break
        best = value
    return best


def delta(h0, h1, role, metric):
    a, b = h0.get(role, {}).get(metric, {}), h1.get(role, {}).get(metric, {})
    return {k: v - a.get(k, 0) for k, v in b.items()}


def p50(buckets):
    """Median from cumulative histogram buckets, interpolated within the bucket; (median, count)."""
    les = sorted((float('inf') if k == '+Inf' else float(k), c) for k, c in buckets.items())
    total = les[-1][1] if les else 0
    if total <= 0:
        return None, 0
    prev_le, prev_c = 0.0, 0.0
    for le, c in les:
        if c >= total / 2:
            if le == float('inf'):
                return prev_le, total
            return prev_le + (le - prev_le) * (total / 2 - prev_c) / max(c - prev_c, 1e-9), total
        prev_le, prev_c = le, c


# --- Per-framework update records ---------------------------------------------------------------------

def miles_updates(result):
    """{update: dict(end, step_s, train_s, wait_s, filtered, samples=[(task, prompt, response, score)])}."""
    perf = re.compile(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) actor_cell0_rank0\] train_metric_utils.py:\d+ - perf (\d+): (\{.*\})')
    rollout = re.compile(r'rollout_executor\] metrics.py:\d+ - perf (\d+): (\{.*\})')
    updates, stale = {}, {}
    for line in open(result / 'learner-train.out', errors='replace'):
        if m := perf.search(line):
            end = datetime.datetime.strptime(m[1], '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc).timestamp()
            p = eval(m[3])  # Miles logs a Python dict literal
            updates[int(m[2]) + 1] = {'end': end, 'step_s': p.get('perf/step_time'), 'train_s': p.get('perf/train_time'),
                                      'wait_s': p.get('perf/train_wait_time'), 'samples': []}
        elif m := rollout.search(line):
            stale[int(m[1]) + 1] = eval(m[2]).get('rollout/fully_async/stale_groups_filtered', 0) or 0
    for line in open(result / 'train-samples.jsonl'):
        s = json.loads(line)
        if (u := s['rollout_id'] + 1) in updates:
            updates[u]['samples'].append((s['domain'], s['prompt_tokens'], s['response_tokens'], s['score']))
    for u, row in updates.items():
        row['filtered'] = stale.get(u, 0)
    return updates


def verl_updates(result, root, base_model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    prompts = {}  # user prompt -> task, from the packaged training data
    for path in sorted((root / 'data').glob('*-train.jsonl*')):  # .jsonl in a campaign, .jsonl.gz in the repo
        task = path.name.split('-train.jsonl')[0]
        if task in TASKS:
            with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
                prompts.update((json.loads(l)['prompt'][-1]['content'].strip(), task) for l in stream)
    logged = {}
    for log in [result / 'learner-train.out', *sorted(result.glob('*trainer-worker.out'))]:
        for line in log.read_text(errors='replace').splitlines() if log.exists() else ():
            if 'timing_s/step:' in line:
                step = int(line.split('step:', 1)[1].split(' ', 1)[0])
                fields = dict(f.strip().rsplit(':', 1) for f in line.split('step:', 1)[1].split(' - ')[1:] if ':' in f)
                logged[step] = {k: float(v) for k, v in fields.items() if re.fullmatch(r'[-+.\deE]+', v.strip())}
    updates, dropped_before = {}, 0
    for path in sorted((result / 'rollouts').glob('*.jsonl'), key=lambda p: int(p.stem)):
        u, m = int(path.stem), logged.get(int(path.stem) + 1, {})
        samples = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            user = row['input'].rsplit('\nuser\n', 1)[-1].rsplit('\nassistant\n', 1)[0].strip()
            samples.append((prompts.get(user), len(tokenizer(row['input'])['input_ids']),
                            len(tokenizer(row['output'])['input_ids']), row['score']))
        dropped = m.get('fully_async/count/dropped_stale_samples', dropped_before)
        updates[u] = {'end': path.stat().st_mtime, 'step_s': m.get('timing_s/step'), 'train_s': m.get('timing_s/update_actor'),
                      'wait_s': m.get('timing_s/gen'), 'filtered': dropped - dropped_before, 'samples': samples,
                      'train_tokens_logged': m.get('perf/total_num_tokens')}
        dropped_before = dropped
    return updates


def main(result, out):
    result = Path(result)
    root, framework, domains, nodes, policy_gpus, teacher_gpus = campaign(result)
    site = json.loads((root / 'site.json').read_text())
    updates = miles_updates(result) if framework == 'miles' else verl_updates(result, root, site['base_model'])
    series = metric_series(result, domains)
    start = json.loads((result / 'start.json').read_text())['time']
    rows, previous = [], start
    for u in sorted(updates):
        r, t0, t1 = updates[u], previous, updates[u]['end']
        window = t1 - t0
        h0, h1 = at(series, t0), at(series, t1)
        samples = r['samples']
        tokens = r.get('train_tokens_logged') or sum(p + q for _, p, q, _ in samples) or None
        rollout_p50, finished = p50(delta(h0, h1, 'policy', 'e2e_request_latency_seconds_bucket'))
        teacher_p50, _ = p50(delta(h0, h1, 'teacher', 'e2e_request_latency_seconds_bucket'))
        generated = sum(delta(h0, h1, 'policy', 'generation_tokens_total').values())
        teacher_tokens = sum(delta(h0, h1, 'teacher', 'prompt_tokens_total').values())
        tok_s_gpu = tokens / r['train_s'] / TRAINER_GPUS if tokens and r['train_s'] else None
        teacher_tok_s_gpu = teacher_tokens / window / teacher_gpus if teacher_tokens else None
        scores = {}
        for task, _, _, score in samples:
            if task and score is not None:
                scores.setdefault(task, []).append(score)
        rows.append({
            'update': u, 'window_s': window, 'step_s': r['step_s'], 'train_s': r['train_s'], 'wait_s': r['wait_s'],
            'train_tokens': tokens, 'train_tok_s_gpu': tok_s_gpu,
            'mfu': tok_s_gpu * 6 * ACTIVE_PARAMS / H100_PEAK_FLOPS if tok_s_gpu else None,
            'prompt_p50': statistics.median(p for _, p, _, _ in samples) if samples else None,
            'response_p50': statistics.median(q for _, _, q, _ in samples) if samples else None,
            'rollout_e2e_p50_s': rollout_p50, 'rollout_requests': finished,
            'rollout_tok_s_gpu': generated / window / policy_gpus if generated else None,
            'filtered': r['filtered'], 'filtered_ratio': r['filtered'] / (r['filtered'] + BATCH),
            'effective_tok_s_gpu_policy': tokens / window / policy_gpus if tokens else None,
            'teacher_p50_s': teacher_p50, 'teacher_tok_s_gpu': teacher_tok_s_gpu,
            'teacher_mfu': teacher_tok_s_gpu * 2 * ACTIVE_PARAMS / H100_PEAK_FLOPS if teacher_tok_s_gpu else None,
            'cost_usd': window * nodes * GPUS_PER_NODE * H100_USD_PER_HOUR / 3600,
            'train_score': {k: statistics.fmean(v) for k, v in sorted(scores.items())},
            'samples': len(samples)})
        previous = t1
    meta = {'result': str(result), 'framework': framework, 'domains': domains, 'nodes': nodes,
            'policy_gpus': policy_gpus, 'teacher_gpus': teacher_gpus, 'active_params': ACTIVE_PARAMS,
            'h100_peak_flops': H100_PEAK_FLOPS, 'h100_usd_per_hour': H100_USD_PER_HOUR}
    Path(out).write_text(json.dumps({'meta': meta, 'updates': rows}, indent=1) + '\n')
    for row in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != 'train_score'})


if __name__ == '__main__':
    main(*sys.argv[1:3])
