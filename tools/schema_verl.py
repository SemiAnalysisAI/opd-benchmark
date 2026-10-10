"""Per-update case-study metrics for a verl run, in the columns of Miles run 535's schema file.

    <campaign>/source/.venv/bin/python tools/schema_verl.py RESULT_DIR OUT.json

Sources, all written by every verl run of this package:
  - `*trainer-worker.out` / `learner-train.out`: verl's console metrics per step (timing, MFU, tokens,
    dropped stale samples). verl logs update N at metric step N + 1 (see tools/report.py).
  - `rollouts/<N>.jsonl`: written at the end of update N, so its modification time ends the update's
    window (it matches the trainer's progress bar to about a second). It holds the trained samples' decoded
    prompt and response, re-tokenized here with the student's tokenizer. Decoding dropped special tokens,
    so prompt lengths are a few tokens short. (verl's TensorBoard writer runs in the trainer's Ray worker,
    which does not get TENSORBOARD_DIR, so the run's tensorboard/ directory stays empty.)
  - `*-capture.jsonl.gz`: vLLM Prometheus histograms and counters of every policy and teacher server,
    scraped every 5 s by shared/capture.py --vllm. A server is a teacher if its model path is under the
    site's teachers directory. Policy requests resumed after a weight sync (partial rollout) count as
    separate requests.
"""
import gzip
import json
from pathlib import Path
import re
import statistics
import sys

TRAINER_GPUS, GPUS_PER_TEACHER, BATCH = 8, 2, 128  # h100x3 and h100x2


def console_metrics(result):
    """{logged step: {metric: value}} from verl's console lines."""
    steps = {}
    logs = [result / 'learner-train.out', *sorted(result.glob('*trainer-worker.out'))]
    for log in logs:
        if not log.exists():
            continue
        for line in log.read_text(errors='replace').splitlines():
            if 'timing_s/step:' not in line:
                continue
            step = int(line.split('step:', 1)[1].split(' ', 1)[0])
            fields = dict(f.strip().rsplit(':', 1) for f in line.split('step:', 1)[1].split(' - ')[1:] if ':' in f)
            steps[step] = {k: float(v) for k, v in fields.items() if re.fullmatch(r'[-+.\deE]+', v.strip())}
    return steps


def update_ends(result):
    """{update: wall time it ended}, from the modification time of verl's per-update rollout dump."""
    return {int(p.stem): p.stat().st_mtime for p in (result / 'rollouts').glob('*.jsonl')}


def histograms(result, teachers_dir):
    """[(time, {'policy'|'teacher': {metric: {le or '': value}}})] summed over servers, per capture row."""
    rows = []
    for path in sorted(result.glob('*-capture.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            try:  # A capture still being written ends mid-stream; keep the rows read so far.
                for line in stream:
                    row = json.loads(line)
                    rows.append((row['time_unix'], row['prometheus']))
            except (EOFError, json.JSONDecodeError):
                pass
    rows.sort(key=lambda r: r[0])
    # Each capture file is one node; keep the latest value per (server, series) and sum over servers.
    latest, out = {}, []
    names = ('vllm:e2e_request_latency_seconds_bucket', 'vllm:prompt_tokens_total', 'vllm:generation_tokens_total',
             'vllm:request_success_total')
    for t, servers in rows:
        for target, text in servers.items():
            if not isinstance(text, str):
                continue
            for line in text.splitlines():
                if not line.startswith(names):
                    continue
                series, value = line.rsplit(' ', 1)
                role = 'teacher' if teachers_dir in series else 'policy'
                metric = series.split('{', 1)[0]
                le = m[1] if (m := re.search(r'le="([^"]+)"', series)) else ''
                latest[(target, series)] = (role, metric, le, float(value))
        total = {}
        for role, metric, le, value in latest.values():
            bucket = total.setdefault(role, {}).setdefault(metric, {})
            bucket[le] = bucket.get(le, 0) + value
        out.append((t, total))
    return out


def at(series, t):
    """The cumulative values at time t (the last capture at or before t)."""
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
    """Median from cumulative histogram buckets, interpolated within the bucket."""
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


def sample_lengths(result, tokenizer):
    """{dump step: [(prompt tokens, response tokens), ...]} from verl's rollout dump."""
    lengths = {}
    for path in sorted((result / 'rollouts').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            lengths.setdefault(row['step'], []).append(
                (len(tokenizer(row['input'])['input_ids']), len(tokenizer(row['output'])['input_ids'])))
    return lengths


def main(result, out):
    result = Path(result)
    site = json.loads((result.parents[1] / 'site.json').read_text())
    experiment = result.parents[1] / 'experiment.json'
    experiment = json.loads(experiment.read_text()) if experiment.exists() else {}
    domains = experiment.get('domains', [1, 2])
    POLICY_GPUS = 6 if experiment.get('layout') == 'h100x2' else 12
    teacher_gpus = GPUS_PER_TEACHER * len(domains)  # 2 for single-teacher OPD, 4 for MOPD
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(site['base_model'])
    metrics, ends = console_metrics(result), update_ends(result)
    hist = histograms(result, site['teachers'])
    lengths = sample_lengths(result, tokenizer)
    start = json.loads((result / 'start.json').read_text())['time']
    rows, previous_end, previous_dropped = [], start, 0
    for update in sorted(ends):
        # verl logs update N's metrics at step N + 1 and never logs the last update's.
        m, t0, t1 = metrics.get(update + 1, {}), previous_end, ends[update]
        h0, h1 = at(hist, t0), at(hist, t1)
        rollout_p50, finished = p50(delta(h0, h1, 'policy', 'vllm:e2e_request_latency_seconds_bucket'))
        teacher_p50, _ = p50(delta(h0, h1, 'teacher', 'vllm:e2e_request_latency_seconds_bucket'))
        teacher_tokens = sum(delta(h0, h1, 'teacher', 'vllm:prompt_tokens_total').values())
        samples = lengths.get(update, [])
        dropped_total = m.get('fully_async/count/dropped_stale_samples', previous_dropped)
        dropped = dropped_total - previous_dropped
        tokens = m.get('perf/total_num_tokens')
        rows.append({
            'update': update, 'step_s': m.get('timing_s/step'), 'train_s': m.get('timing_s/update_actor'),
            'wait_s': m.get('timing_s/gen'),
            'train_tok_s_gpu': tokens / m['timing_s/update_actor'] / TRAINER_GPUS
            if tokens and m.get('timing_s/update_actor') else None,
            'mfu': m.get('perf/mfu/actor'),
            'prompt_p50': statistics.median(p for p, _ in samples) if samples else None,
            'response_p50': statistics.median(r for _, r in samples) if samples else None,
            'rollout_e2e_p50_s': rollout_p50, 'rollout_requests_finished': finished,
            'stale_filtered': dropped, 'filtered_ratio': dropped / (dropped + BATCH),
            'effective_tok_s_gpu_policy': sum(p + r for p, r in samples) / (t1 - t0) / POLICY_GPUS if samples else None,
            'teacher_p50_s': teacher_p50,
            'teacher_tok_s_gpu': teacher_tokens / (t1 - t0) / teacher_gpus if teacher_tokens else None,
            'window_s': t1 - t0, 'samples_dumped': len(samples)})
        previous_end, previous_dropped = t1, dropped_total
    Path(out).write_text(json.dumps(rows, indent=1) + '\n')
    for row in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == '__main__':
    main(*sys.argv[1:3])
