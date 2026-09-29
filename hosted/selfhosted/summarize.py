"""Summarize one self-hosted client run (sft.py, rl.py, opd.py or mopd.py) for PostTrainingX.

    python hosted/selfhosted/summarize.py RUNS_DIR/NAME.window.json OUTPUT_DIR --telemetry-code DIR

NAME.window.json (written by client.sh) names the server directory and the run's UTC window. OUTPUT_DIR is the run
directory the client wrote. The summary goes to RUNS_DIR/NAME.summary.json:

- framework, workload, server description (server.json), exit code, window;
- topology: nodes, physical GPUs, and trainer and inference nodes (server.json `node_roles`, else inferred from GPU
  memory: an inference node's engines reserve at least 80% of every GPU before the run);
- the protocol (config.json) and one record per optimizer step with its UTC completion time, tokens and timings;
- the throughput window: tokens trained in steps 2..N over the time from step 1's completion to step N's, per
  second, per physical GPU (the whole allocation) and per trainer GPU. Step 1 is excluded because it carries
  compilation and engine start. RL and OPD tokens are prompt plus response tokens of every trained sequence; SFT
  tokens are prompt plus completion tokens;
- quality: the evaluations (opd, mopd, rl) or the teacher benchmark (sft);
- telemetry cut to the window and summarized by the ClusterMAX collector's own functions (GPU utilization, power,
  memory and clocks by role; NVLink, InfiniBand and Ethernet bytes by role).
"""
import argparse
import csv
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import statistics
import tempfile

GPUS_PER_NODE = 8


def utc(text):
    return datetime.fromisoformat(text.replace('Z', '+00:00'))


def load_collector(directory):
    """The ClusterMAX miles-terminal-lego collect_results.py, for its telemetry summarizers."""
    spec = importlib.util.spec_from_file_location('clustermax_collect', Path(directory)/'collect_results.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cut_csv(source, target, start, end):
    """Copy the rows of a telemetry CSV whose collector_time_utc is inside [start, end]."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline='') as f, target.open('w', newline='') as g:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(g, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            try:
                stamp = utc(row['collector_time_utc'])
            except (KeyError, ValueError):
                continue
            if start <= stamp <= end:
                writer.writerow(row)


def server_roles(server_dir):
    """Trainer and inference nodes from the server's own logs: the SkyRL server log names its vLLM servers' URLs."""
    ips = json.loads((server_dir/'node-ips.json').read_text()) if (server_dir/'node-ips.json').is_file() else {}
    log = server_dir/'logs'/'server.log'
    if not ips or not log.is_file():
        return None
    import re
    text = log.read_text(errors='replace').replace('\n', '')
    match = re.search(r"server_urls=\[([^\]]*)\]", re.sub(r'\s+', '', text))
    if not match:
        return None
    engine_ips = set(re.findall(r'//([0-9.]+):', match.group(1)))
    inference = sorted(n for n, ip in ips.items() if ip in engine_ips)
    return {'trainer': sorted(n for n in ips if n not in inference), 'inference': inference}


def infer_roles(gpu_dir, start):
    """Inference nodes are those whose every GPU already holds at least 80% of its memory at the run's start."""
    roles = {'trainer': [], 'inference': []}
    for path in sorted(gpu_dir.glob('*.csv')):
        with path.open(newline='') as f:
            rows = [r for r in csv.DictReader(f) if r.get('collector_time_utc') and utc(r['collector_time_utc']) >= start]
        first = rows[:GPUS_PER_NODE]
        if not first:
            continue
        full = all(float(r['memory.used']) >= 0.8 * float(r['memory.total']) for r in first)
        roles['inference' if full else 'trainer'].append(path.stem)
    return roles


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def step_records(output, workload):
    """(step, completion time, tokens, record) per optimizer step."""
    rows = []
    if workload == 'sft':
        for m in read_jsonl(output/'metrics.jsonl'):
            rows.append({'step': m['step'], 'utc': m['utc'], 'tokens': m['prompt_tokens'] + m['completion_tokens'],
                         'prompt_tokens': m['prompt_tokens'], 'completion_tokens': m['completion_tokens'],
                         'sequences': m['sequences'], 'train_nll': m['train_nll'], 'learning_rate': m['learning_rate'],
                         'interval_seconds': m['interval_seconds'], 'step_seconds': m['step_seconds']})
        return rows
    completed = {e['update']: e['utc'] for e in read_jsonl(output/'events.jsonl') if e['event'] == 'update_complete'}
    for m in read_jsonl(output/'metrics.jsonl'):
        row = {'step': m['update'], 'utc': completed.get(m['update']), 'tokens': m['prompt_tokens'] + m['response_tokens'],
               'prompt_tokens': m['prompt_tokens'], 'response_tokens': m['response_tokens'], 'responses': m['responses'],
               'policy_lag': m['policy_lag'], 'truncated': m['truncated'], 'timing': m['timing'],
               'throughput': m['throughput'], 'by_domain': m['by_domain'],
               'train_sampler_logprob_delta_abs_mean': m.get('train_sampler_logprob_delta_abs_mean')}
        if 'reward' in m:
            row['reward'] = m['reward']
        rows.append(row)
    return rows


def throughput(steps, physical_gpus, trainer_gpus):
    timed = [s for s in steps if s.get('utc')]
    if len(timed) < 2:
        return None
    first, last = timed[0], timed[-1]
    seconds = (utc(last['utc']) - utc(first['utc'])).total_seconds()
    tokens = sum(s['tokens'] for s in timed[1:])
    rate = tokens / seconds
    return {'first_step': first['step'], 'last_step': last['step'], 'seconds': seconds, 'tokens': tokens,
            'tokens_per_second': rate, 'tokens_per_second_per_physical_gpu': rate / physical_gpus,
            'tokens_per_second_per_trainer_gpu': rate / trainer_gpus if trainer_gpus else None,
            'steps_per_hour': 3600 * (len(timed) - 1) / seconds}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('window', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--telemetry-code', type=Path, required=True, help='Directory holding ClusterMAX collect_results.py')
    args = p.parse_args()
    window = json.loads(args.window.read_text())
    server_dir = Path(window['server'])
    server = json.loads((server_dir/'server.json').read_text())
    config = json.loads((args.output/'config.json').read_text()) if (args.output/'config.json').is_file() else {}
    workload = 'sft' if config.get('mode') == 'train' else config.get('kind')
    start, end = utc(window['start_utc']), utc(window['end_utc'])

    collector = load_collector(args.telemetry_code)
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        for source in (server_dir/'telemetry').rglob('*.csv'):
            relative = source.relative_to(server_dir)
            if relative.parts[2:3] == ('ib-perfquery',):  # perfquery samples replace fabric_monitor.sh's empty IB rows.
                relative = Path('telemetry', 'fabric', 'ib', source.name)
            elif relative.parts[2:3] == ('ib',) and (server_dir/'telemetry'/'fabric'/'ib-perfquery').is_dir():
                continue
            cut_csv(source, scratch/relative, start, end)
        roles, role_source = server.get('node_roles'), 'server.json'
        if not roles:
            roles, role_source = server_roles(server_dir), 'server log'
        if not roles:
            roles, role_source = infer_roles(scratch/'telemetry'/'gpu', start), 'gpu-memory'
        (scratch/'provenance').mkdir()
        (scratch/'provenance'/'node-roles.json').write_text(json.dumps(roles))
        role_sets = collector.node_roles(scratch)
        telemetry = {'gpu': collector.summarize_gpu(scratch, role_sets),
                     'nvlink': collector.summarize_nvlink(scratch, role_sets),
                     'ib': collector.summarize_ib(scratch, role_sets),
                     'net': collector.summarize_net(scratch, role_sets)}

    nodes = server['nodes']
    # A colocated server (verl-tinker's hybrid engine) trains and samples on the same 'shared' nodes.
    trainer_gpus = GPUS_PER_NODE * len(roles.get('trainer', []) + roles.get('shared', []))
    steps = step_records(args.output, workload)
    intervals = [(utc(b['utc']) - utc(a['utc'])).total_seconds() for a, b in zip(steps, steps[1:]) if a.get('utc') and b.get('utc')]
    quality = ({'benchmark': json.loads((args.output/'benchmark.json').read_text())} if (args.output/'benchmark.json').is_file()
               else {'evaluations': read_jsonl(args.output/'evaluations.jsonl')})
    summary = {
        'schema_version': 1, 'framework': server['framework'], 'workload': workload, 'name': window['name'],
        'exit_code': window['exit_code'], 'complete': window['exit_code'] == 0 and not (args.output/'error.txt').exists(),
        'window': {'start_utc': window['start_utc'], 'end_utc': window['end_utc'], 'seconds': (end - start).total_seconds()},
        'server': server, 'command': window['command'], 'output_dir': str(args.output),
        'topology': {'nodes': nodes, 'gpus_per_node': GPUS_PER_NODE, 'physical_gpus': GPUS_PER_NODE * len(nodes),
                     'node_roles': roles, 'role_source': role_source,
                     'trainer_gpus': trainer_gpus,
                     'inference_gpus': GPUS_PER_NODE * len(roles.get('inference', []) + roles.get('shared', []))},
        'protocol': config, 'steps': steps,
        'step_seconds': ({'median': statistics.median(intervals), 'mean': statistics.mean(intervals),
                          'min': min(intervals), 'max': max(intervals), 'basis': 'steps 2..N completion intervals'}
                         if intervals else None),
        'throughput_window': throughput(steps, GPUS_PER_NODE * len(nodes), trainer_gpus),
        'quality': quality, 'telemetry': telemetry,
    }
    target = args.window.with_name(args.window.name.replace('.window.json', '.summary.json'))
    target.write_text(json.dumps(summary, indent=2, default=str))
    print(target)


if __name__ == '__main__':
    main()
