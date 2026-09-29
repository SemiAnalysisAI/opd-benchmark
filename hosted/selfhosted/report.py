"""Compare the self-hosted Tinker-compatible frameworks from their run summaries (summarize.py).

    python hosted/selfhosted/report.py $CAMPAIGN --output $CAMPAIGN/report

Reads $CAMPAIGN/<framework>/runs/*.summary.json for every framework directory and writes comparison.json (one row
per framework and run) and REPORT.md (the tables below). Throughput is the summary's window: tokens trained in steps
2..N over the time from step 1 to step N, per physical GPU of the 32-GPU allocation. The quality columns are the
teacher benchmark for SFT (avg@3 on the first 100 dev problems of the teacher's domain) and the evaluations at
updates 0 and last for RL and MOPD (avg@3 on the first 100 dev problems of each domain, temperature 1).
"""
import argparse
import json
from pathlib import Path

DOMAINS = ('caesar_cipher', 'simple_geometry')


def row(summary):
    t = summary.get('throughput_window') or {}
    gpu = summary['telemetry']['gpu']
    ib = summary['telemetry']['ib'].get('by_role', {})
    nvlink = summary['telemetry']['nvlink'].get('by_role', {})
    quality = {}
    if 'benchmark' in summary['quality']:
        bench = summary['quality']['benchmark']
        quality = {d: {'final': bench[d]['accuracy']} for d in DOMAINS if d in bench}
    else:
        evals = summary['quality']['evaluations']
        if evals:
            quality = {d: {'initial': evals[0][d]['accuracy'], 'final': evals[-1][d]['accuracy']} for d in DOMAINS if d in evals[0]}
    roles = summary['topology']['node_roles']
    trained_role = 'trainer' if 'trainer' in gpu else 'shared'
    sampling_role = 'inference' if 'inference' in gpu else 'shared'
    return {
        'framework': summary['framework'], 'run': summary['name'], 'workload': summary['workload'],
        'complete': summary['complete'], 'steps': len(summary['steps']), 'wall_seconds': summary['window']['seconds'],
        'median_step_seconds': (summary.get('step_seconds') or {}).get('median'),
        'tokens_per_second': t.get('tokens_per_second'),
        'tokens_per_second_per_physical_gpu': t.get('tokens_per_second_per_physical_gpu'),
        'tokens_per_second_per_trainer_gpu': t.get('tokens_per_second_per_trainer_gpu'),
        'trainer_gpus': summary['topology']['trainer_gpus'], 'inference_gpus': summary['topology']['inference_gpus'],
        'node_roles': roles,
        'trainer_gpu_utilization_pct': gpu.get(trained_role, {}).get('utilization_mean_pct'),
        'inference_gpu_utilization_pct': gpu.get(sampling_role, {}).get('utilization_mean_pct'),
        'all_gpu_power_watts': gpu.get('all', {}).get('power_mean_watts'),
        'nvlink_tx_gigabytes_per_second': nvlink.get('all', {}).get('tx_mean_gigabytes_per_second'),
        'ib_xmit_gigabytes_per_second': ib.get('all', {}).get('xmit_mean_gigabytes_per_second'),
        'quality': quality,
    }


def fmt(value, digits=0, pct=False):
    if value is None:
        return 'n/a'
    return f'{100 * value:.1f}%' if pct else f'{value:,.{digits}f}'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('campaign', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = [row(json.loads(path.read_text())) for path in sorted(args.campaign.glob('*/runs/*.summary.json'))]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'comparison.json').write_text(json.dumps(rows, indent=2))
    lines = ['# Self-hosted Tinker-compatible endpoints: SFT, RL and MOPD', '',
             'Every run used the same client code (hosted/opd) and 32 B200 GPUs. Throughput is steps 2..N; see '
             'hosted/selfhosted/summarize.py for the definitions.', '',
             '| Framework | Run | Complete | Steps | Wall (min) | Median step (s) | Tokens/s | Tokens/s/GPU | Trainer GPU util | '
             'Sampler GPU util | NVLink TX GB/s | IB TX GB/s |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['framework']} | {r['run']} | {'yes' if r['complete'] else 'no'} | {r['steps']} | "
                     f"{fmt(r['wall_seconds'] / 60, 1)} | {fmt(r['median_step_seconds'], 1)} | {fmt(r['tokens_per_second'])} | "
                     f"{fmt(r['tokens_per_second_per_physical_gpu'])} | {fmt(r['trainer_gpu_utilization_pct'], 1)} | "
                     f"{fmt(r['inference_gpu_utilization_pct'], 1)} | {fmt(r['nvlink_tx_gigabytes_per_second'], 1)} | "
                     f"{fmt(r['ib_xmit_gigabytes_per_second'], 2)} |")
    lines += ['', '## Quality (avg@3)', '', '| Framework | Run | caesar_cipher | simple_geometry |', '|---|---|---|---|']
    for r in rows:
        cells = []
        for d in DOMAINS:
            q = r['quality'].get(d)
            cells.append('' if not q else (f"{fmt(q.get('initial'), pct=True)} -> {fmt(q['final'], pct=True)}"
                                           if 'initial' in q else fmt(q['final'], pct=True)))
        lines.append(f"| {r['framework']} | {r['run']} | {cells[0]} | {cells[1]} |")
    (args.output/'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(args.output/'REPORT.md')


if __name__ == '__main__':
    main()
