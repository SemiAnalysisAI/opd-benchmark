"""Summarize the telemetry of finished campaign runs.

    python3 tools/report.py RESULT_DIR [RESULT_DIR ...] [--output report.json]

Reads each run's `start.json`, `end.json` and `<role>-capture.jsonl.gz` (written by
`shared/capture.py`) and reports, per role: GPU utilization, memory and power for every
GPU, energy, host CPU utilization, network traffic, and the token throughput of every
captured SGLang or vLLM engine (from its Prometheus token counters). A capture cut off by a
cancelled run is read up to its last complete row.

It also reads each framework's own per-update metrics (the Miles and verl driver logs,
Prime-RL's `monitors/file/metrics.jsonl`) into the common fields of `UPDATE_FIELDS`.
Standard library only.
"""
import argparse
import ast
import gzip
import json
from pathlib import Path
import re
import statistics
import zlib

TOKEN_COUNTERS = {'generation': ('sglang:generation_tokens_total', 'vllm:generation_tokens_total'),
                  'prompt': ('sglang:prompt_tokens_total', 'vllm:prompt_tokens_total')}


# Common per-update fields and each framework's metric for them.
UPDATE_FIELDS = {
    'step_s': {'miles': 'perf/step_time', 'slime': 'perf/step_time', 'verl': 'timing_s/step', 'prime-rl': 'time/step'},
    'train_s': {'miles': 'perf/train_time', 'slime': 'perf/train_time', 'verl': 'timing_s/update_actor',
                'prime-rl': 'time/forward_backward'},
    'wait_for_rollouts_s': {'miles': 'perf/train_wait_time', 'slime': 'perf/train_wait_time', 'verl': 'timing_s/gen',
                            'prime-rl': 'time/wait_for_batch'},
    'weight_sync_s': {'miles': 'perf/update_weights_time', 'slime': 'perf/update_weights_time',
                      'verl': 'timing_s/timing_s/param_sync', 'prime-rl': 'time/broadcast_weights'},
    'mfu': {'miles': 'perf/actor_train_mfu', 'slime': 'perf/actor_train_mfu', 'verl': 'perf/mfu/actor',
            'prime-rl': 'perf/mfu'},
    'response_tokens_mean': {'miles': 'rollout/response_lengths', 'slime': 'rollout/response_lengths',
                             'verl': 'response_length/mean', 'prime-rl': 'train/agg/all/num_output_tokens/mean'},
    'reverse_kl': {'miles': 'train/opd_reverse_kl', 'slime': 'rollout/opd_reverse_kl', 'verl': 'actor/distillation/loss',
                   'prime-rl': 'ref_kl/mean'},
    # Miles logs no verifier score or stale-drop count for OPD training rollouts; Slime's raw reward is the
    # package's task score, and its one-rollout-ahead mode drops nothing.
    'train_score': {'miles': 'rollout/raw_reward', 'slime': 'rollout/raw_reward', 'verl': 'critic/score/mean',
                    'prime-rl': 'train/agg/all/agent/reward/mean'},
    'dropped_stale': {'verl': 'fully_async/count/dropped_stale_samples',
                      'prime-rl': 'off_policy/dropped'},
}
MILES_METRICS = re.compile(r'rank0\] \S+ - (?:perf|rollout|step) (\d+): (\{.*\})\s*$')
SLIME_METRICS = re.compile(r'\] \S+\.py:\d+ - (?:perf|rollout|step) (\d+): (\{.*\})\s*$')
TENSOR = re.compile(r"tensor\(\[?([-+\d.eE]+)\]?(?:, device='[^']*')?\)")
B200_PEAK_TFLOPS = 2250  # Dense bf16 peak Miles records as perf/mfu_peak_tflops; Slime logs TFLOPS only.


def framework_updates(result, framework):
    """{update: {metric: value}} from the framework's own logs."""
    updates = {}
    if framework in ('miles', 'slime'):
        pattern = MILES_METRICS if framework == 'miles' else SLIME_METRICS
        for line in (result / 'learner-train.out').read_text(errors='replace').splitlines():
            if match := pattern.search(line):
                # Slime logs some one-element tensors, e.g. tensor([0.43], device='cuda:0').
                values = ast.literal_eval(TENSOR.sub(r'\1', match[2]))
                updates.setdefault(int(match[1]), {}).update(
                    {k: v for k, v in values.items() if isinstance(v, (int, float))})
        if framework == 'slime':
            for values in updates.values():
                if 'perf/actor_train_tflops' in values:
                    values['perf/actor_train_mfu'] = values['perf/actor_train_tflops'] / B200_PEAK_TFLOPS
    elif framework == 'verl':
        for line in (result / 'learner-train.out').read_text(errors='replace').splitlines():
            if 'timing_s/step:' in line:
                fields = dict(f.strip().rsplit(':', 1) for f in line.split('step:', 1)[1].split(' - ')[1:] if ':' in f)
                step = int(line.split('step:', 1)[1].split(' ', 1)[0])
                updates[step] = {k: float(v) for k, v in fields.items() if re.fullmatch(r'[-+.\deE]+', v)}
    elif framework == 'prime-rl':
        path = result / 'monitors' / 'file' / 'metrics.jsonl'
        for line in path.read_text().splitlines() if path.exists() else ():
            row = json.loads(line)
            if row.get('step') is not None:
                updates.setdefault(row['step'], {}).update(row)
    return updates


def update_summary(result, framework):
    updates = framework_updates(result, framework)
    rows = [{field: values.get(names.get(framework)) for field, names in UPDATE_FIELDS.items()}
            for _, values in sorted(updates.items())]
    if framework == 'prime-rl':  # Prime-RL reports MFU in percent; the others as a fraction.
        for row in rows:
            row['mfu'] = row['mfu'] / 100 if isinstance(row['mfu'], (int, float)) else row['mfu']
    rows = [r for r in rows if r['step_s'] is not None]
    if not rows:
        return None
    if framework == 'miles' and not any(r['train_score'] for r in rows):
        # Before the package's reward hook set metadata['raw_reward'], Miles logged its OPD reward, a constant 0.
        for row in rows:
            row['train_score'] = None

    def column(field):
        return [r[field] for r in rows if isinstance(r[field], (int, float))]
    summary = {'updates': len(rows), 'per_update': rows}
    for field in ('step_s', 'train_s', 'wait_for_rollouts_s', 'weight_sync_s'):
        if values := column(field):
            summary[f'{field}_median'] = statistics.median(values)
            summary[f'{field}_total'] = sum(values)
    for field in ('mfu', 'response_tokens_mean'):
        if values := column(field):
            summary[f'{field}_mean'] = statistics.fmean(values)
    for field in ('reverse_kl', 'train_score'):
        if values := column(field):
            summary[f'{field}_first'], summary[f'{field}_last'] = values[0], values[-1]
    if values := column('dropped_stale'):
        summary['dropped_stale_total'] = sum(values)
    return summary


def capture_rows(path):
    rows = []
    try:
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                rows.append(json.loads(line))
    except (EOFError, zlib.error, json.JSONDecodeError):
        pass  # The run was stopped while the capture was writing.
    return rows


def gpu_summary(rows):
    samples = {}
    for row in rows:
        if not isinstance(row['gpu_csv'], str):
            continue
        for line in row['gpu_csv'].splitlines()[1:]:
            fields = [f.strip() for f in line.split(',')]
            index, util, memory, power = int(fields[1]), fields[3], fields[5], fields[7]
            if '[N/A]' in (util, memory, power):
                continue
            samples.setdefault(index, []).append((row['time_unix'], float(util.split()[0]),
                                                  float(memory.split()[0]) / 1024, float(power.split()[0])))
    summary = {}
    for index, values in sorted(samples.items()):
        energy = sum((b[0] - a[0]) * a[3] for a, b in zip(values, values[1:])) / 3.6e6
        summary[index] = {'utilization_mean_pct': statistics.fmean(v[1] for v in values),
                          'busy_fraction': statistics.fmean(v[1] > 0 for v in values),
                          'memory_peak_gib': max(v[2] for v in values),
                          'power_mean_w': statistics.fmean(v[3] for v in values), 'energy_kwh': energy}
    return summary


def cpu_utilization(rows):
    def totals(row):
        fields = [int(x) for x in row['stat'].splitlines()[0].split()[1:]]
        return sum(fields), fields[3] + fields[4]  # All jiffies; idle + iowait.
    rows = [r for r in rows if isinstance(r['stat'], str)]
    if len(rows) < 2:
        return None
    (total_a, idle_a), (total_b, idle_b) = totals(rows[0]), totals(rows[-1])
    return 100 * (1 - (idle_b - idle_a) / (total_b - total_a))


def network(rows):
    def counters(row):
        rx = tx = 0
        for line in row['net/dev'].splitlines()[2:]:
            name, data = line.split(':', 1)
            if name.strip() != 'lo':
                fields = [int(x) for x in data.split()]
                rx, tx = rx + fields[0], tx + fields[8]
        return rx, tx
    rows = [r for r in rows if isinstance(r['net/dev'], str)]
    if len(rows) < 2:
        return None
    (rx_a, tx_a), (rx_b, tx_b) = counters(rows[0]), counters(rows[-1])
    seconds = rows[-1]['time_unix'] - rows[0]['time_unix']
    return {'rx_gb': (rx_b - rx_a) / 1e9, 'tx_gb': (tx_b - tx_a) / 1e9,
            'rx_gbit_s': 8 * (rx_b - rx_a) / 1e9 / seconds, 'tx_gbit_s': 8 * (tx_b - tx_a) / 1e9 / seconds}


def counter(text, names):
    """Sum of every sample of the named Prometheus counters, or None if absent."""
    values = [float(line.rsplit(' ', 1)[1]) for line in text.splitlines()
              if line.split('{')[0].split(' ')[0] in names]
    return sum(values) if values else None


def engine_throughput(rows):
    """Tokens per second of each captured engine, between its first and last successful scrape."""
    engines = {}
    for row in rows:
        for target, text in row['prometheus'].items():
            if isinstance(text, str):
                engines.setdefault(target, []).append((row['time_unix'], text))
    summary = {}
    for target, scrapes in sorted(engines.items()):
        (t0, first), (t1, last) = scrapes[0], scrapes[-1]
        entry = {'seconds': t1 - t0}
        for kind, names in TOKEN_COUNTERS.items():
            # A labeled counter first appears on its first increment, so absent means zero.
            a, b = counter(first, names) or 0.0, counter(last, names)
            if b is not None and t1 > t0:
                entry[f'{kind}_tokens'] = b - a
                entry[f'{kind}_tokens_per_s'] = (b - a) / (t1 - t0)
        summary[target] = entry
    return summary


def summarize(result):
    result = Path(result)
    start = json.loads((result / 'start.json').read_text())
    end = json.loads((result / 'end.json').read_text()) if (result / 'end.json').exists() else {}
    framework = result.name.rsplit('-', 1)[0]
    report = {'run': result.name, 'job_id': start.get('job_id'), 'nodes': start.get('nodes'),
              'exit_code': end.get('exit_code'), 'wall_seconds': end['time'] - start['time'] if end else None,
              'updates': update_summary(result, framework), 'roles': {}}
    for path in sorted(result.glob('*-capture.jsonl.gz')):
        rows = capture_rows(path)
        if not rows:
            continue
        role = path.name[:-len('-capture.jsonl.gz')]
        gpus = gpu_summary(rows)
        report['roles'][role] = {
            'samples': len(rows), 'seconds': rows[-1]['time_unix'] - rows[0]['time_unix'],
            'gpus': gpus, 'gpu_energy_kwh': sum(g['energy_kwh'] for g in gpus.values()),
            'cpu_utilization_pct': cpu_utilization(rows), 'network': network(rows),
            'engines': engine_throughput(rows)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    reports = [summarize(r) for r in args.results]
    text = json.dumps(reports, indent=2)
    if args.output:
        args.output.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
