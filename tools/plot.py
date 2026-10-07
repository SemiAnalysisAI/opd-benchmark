"""Plot one finished (or running) Miles campaign run: timing, scores, lengths, GPUs and engines.

    python3 tools/plot.py RESULT_DIR OUTPUT_DIR

Reads the files the run already writes (no extra instrumentation): the training log
(per-update metrics), `train-samples.jsonl` and `train-reward.jsonl` (frameworks/miles/opd_reward.py),
`teacher-latency-*.jsonl`, and every `<role>-capture.jsonl.gz` (shared/capture.py). On the
generation nodes GPUs 0-1 serve the teachers and GPUs 2-7 the policy (both layouts in
shared/recipe.py); every learner GPU trains. Writes `dashboard.png` and one PNG per panel.
Needs matplotlib.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report  # noqa: E402

# Reference categorical palette (light mode), in fixed slot order.
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
ROLE_COLOR = {'trainer': BLUE, 'policy': ORANGE, 'teacher': AQUA}
DOMAIN_COLOR = {'caesar_cipher': BLUE, 'simple_geometry': ORANGE}
TEACHER_GPUS = (0, 1)


def style(ax, title, ylabel, xlabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=11, color=INK, fontweight='bold')
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8)
    if xlabel == 'update':
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)


def legend(ax):
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)


def gpu_series(result, start):
    """{role: [(minutes, mean util %, mean memory GiB, total power W)]} with GPUs mapped to trainer/policy/teacher."""
    series = {}
    for path in sorted(result.glob('*-capture.jsonl.gz')):
        node_role = path.name[:-len('-capture.jsonl.gz')]
        for row in report.capture_rows(path):
            if not isinstance(row['gpu_csv'], str):
                continue
            groups = {}
            for line in row['gpu_csv'].splitlines()[1:]:
                f = [x.strip() for x in line.split(',')]
                if '[N/A]' in (f[3], f[5], f[7]):
                    continue
                index = int(f[1])
                role = 'trainer' if node_role == 'learner' else 'teacher' if index in TEACHER_GPUS else 'policy'
                groups.setdefault(role, []).append((float(f[3].split()[0]), float(f[5].split()[0]) / 1024,
                                                    float(f[7].split()[0])))
            for role, values in groups.items():
                series.setdefault(role, []).append(((row['time_unix'] - start) / 60, values))
    merged = {}
    for role, rows in series.items():
        # Rows from several nodes: bucket by 5 s so a role's GPUs on two nodes average together.
        buckets = {}
        for minutes, values in rows:
            buckets.setdefault(round(minutes * 12), []).extend(values)
        merged[role] = [(k / 12, statistics.fmean(v[0] for v in vs), statistics.fmean(v[1] for v in vs),
                         sum(v[2] for v in vs)) for k, vs in sorted(buckets.items())]
    return merged


def engine_rates(result, start):
    """{'policy'|'teacher': [(minutes, tokens/s summed over engines)]}: policy generation, teacher prefill."""
    kinds = {'learner': ('policy', report.TOKEN_COUNTERS['generation']),
             'generation': ('teacher', report.TOKEN_COUNTERS['prompt'])}
    rates = {}
    for path in sorted(result.glob('*-capture.jsonl.gz')):
        node_role = path.name[:-len('-capture.jsonl.gz')]
        kind, names = kinds['learner' if node_role == 'learner' else 'generation']
        previous = {}
        for row in report.capture_rows(path):
            for target, text in row['prometheus'].items():
                if not isinstance(text, str):
                    continue
                value = report.counter(text, names) or 0.0
                if target in previous:
                    t0, v0 = previous[target]
                    if row['time_unix'] > t0 and value >= v0:
                        bucket = round((row['time_unix'] - start) / 60 * 2) / 2  # 30 s buckets
                        rates.setdefault(kind, {}).setdefault(bucket, {}).setdefault(target, []).append(
                            (value - v0) / (row['time_unix'] - t0))
                previous[target] = (row['time_unix'], value)
    return {kind: [(b, sum(statistics.fmean(v) for v in targets.values())) for b, targets in sorted(by.items())]
            for kind, by in rates.items()}


def jsonl(paths):
    return [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('result', type=Path, help='run result directory (<campaign>/results/<run>)')
    parser.add_argument('output', type=Path, help='directory for the PNG files')
    args = parser.parse_args()
    result, out = args.result, args.output
    out.mkdir(parents=True, exist_ok=True)
    start = json.loads((result / 'start.json').read_text())['time']
    framework = result.name.rsplit('-', 1)[0]
    updates = report.framework_updates(result, framework) if (result / 'learner-train.out').exists() else {}

    def field(name):
        return report.UPDATE_FIELDS[name][framework]
    steps = sorted(u for u, v in updates.items() if field('step_s') in v)
    samples = jsonl([result / 'train-samples.jsonl'] if (result / 'train-samples.jsonl').exists() else [])
    latency = jsonl(sorted(result.glob('teacher-latency-*.jsonl')))
    gpus, engines = gpu_series(result, start), engine_rates(result, start)
    panels = []

    def panel(name, draw):
        panels.append((name, draw))

    def timing(ax):
        x = [s + 1 for s in steps]
        bottom = [0.0] * len(steps)
        for key, label, color in (('train_s', 'Train', BLUE), ('wait_for_rollouts_s', 'Wait for rollouts', ORANGE),
                                  ('weight_sync_s', 'Weight sync', AQUA)):
            values = [updates[s].get(field(key)) or 0.0 for s in steps]
            ax.bar(x, values, bottom=bottom, color=color, label=label, width=0.8, edgecolor=SURFACE, linewidth=1)
            bottom = [b + v for b, v in zip(bottom, values)]
        style(ax, 'Seconds per update', 'seconds', 'update')
        legend(ax)
    panel('timing', timing)

    def score(ax):
        rows = jsonl([result / 'train-reward.jsonl']) if (result / 'train-reward.jsonl').exists() else []
        for domain, color in DOMAIN_COLOR.items():
            pts = [(r['update'], r['domains'][domain]['score']) for r in rows if domain in r['domains']]
            if pts:
                ax.plot(*zip(*pts), color=color, linewidth=2, marker='o', markersize=4, label=domain)
        style(ax, 'Train-batch verifier score', 'mean score', 'update')
        ax.set_ylim(0, 1)
        legend(ax)
    panel('score', score)

    def lengths(ax):
        for domain, color in DOMAIN_COLOR.items():
            by = {}
            for s in samples:
                if s['domain'] == domain:
                    by.setdefault(s['rollout_id'] + 1, []).append(s['response_tokens'])
            if by:
                x = sorted(by)
                ax.plot(x, [statistics.fmean(by[u]) for u in x], color=color, linewidth=2, label=f'{domain} mean')
                ax.plot(x, [sorted(by[u])[int(0.9 * (len(by[u]) - 1))] for u in x], color=color, linewidth=1.2,
                        linestyle='--', label=f'{domain} p90')
        if not samples and steps:  # Older runs: the batch mean only.
            ax.plot([s + 1 for s in steps], [updates[s].get(field('response_tokens_mean')) for s in steps],
                    color=BLUE, linewidth=2, label='all tasks mean')
        style(ax, 'Trained response length', 'tokens', 'update')
        legend(ax)
    panel('response_length', lengths)

    def length_hist(ax):
        for domain, color in DOMAIN_COLOR.items():
            values = [s['response_tokens'] for s in samples if s['domain'] == domain]
            if values:
                ax.hist(values, bins=40, range=(0, 30720), color=color, alpha=0.55, label=domain,
                        edgecolor=SURFACE, linewidth=0.5)
        style(ax, 'Response length distribution (all updates)', 'samples', 'tokens')
        legend(ax)
    panel('response_length_hist', length_hist)

    def truncation(ax):
        by = {}
        for s in samples:
            by.setdefault(s['rollout_id'] + 1, []).append(s['status'] == 'truncated')
        if by:
            x = sorted(by)
            ax.plot(x, [100 * statistics.fmean(by[u]) for u in x], color=BLUE, linewidth=2, marker='o', markersize=4)
        style(ax, 'Truncated at the response limit', '% of trained samples', 'update')
    panel('truncation', truncation)

    def kl(ax):
        pts = [(s + 1, updates[s][field('reverse_kl')]) for s in steps if field('reverse_kl') in updates[s]]
        if pts:
            ax.plot(*zip(*pts), color=BLUE, linewidth=2, marker='o', markersize=4)
        style(ax, 'Reverse KL to the teacher (train batch)', 'nats / token', 'update')
    panel('reverse_kl', kl)

    for metric, index, title, unit in (('gpu_util', 1, 'GPU utilization by role', 'mean %'),
                                       ('gpu_memory', 2, 'GPU memory by role', 'mean GiB per GPU'),
                                       ('gpu_power', 3, 'GPU power by role', 'total W')):
        def draw(ax, index=index, title=title, unit=unit):
            for role, color in ROLE_COLOR.items():
                if role in gpus:
                    ax.plot([r[0] for r in gpus[role]], [r[index] for r in gpus[role]], color=color,
                            linewidth=1.2, label=role)
            style(ax, title, unit, 'minutes since start')
            legend(ax)
        panel(metric, draw)

    def policy_throughput(ax):
        if 'policy' in engines:
            ax.plot(*zip(*engines['policy']), color=ORANGE, linewidth=1.2)
        style(ax, 'Policy generation throughput (all engines)', 'tokens / s', 'minutes since start')
    panel('policy_throughput', policy_throughput)

    def teacher_throughput(ax):
        if 'teacher' in engines:
            ax.plot(*zip(*engines['teacher']), color=AQUA, linewidth=1.2)
        style(ax, 'Teacher prefill throughput (all teachers)', 'tokens / s', 'minutes since start')
    panel('teacher_throughput', teacher_throughput)

    def teacher_latency(ax):
        for domain, color in DOMAIN_COLOR.items():
            pts = [(r['total_tokens'], r['seconds']) for r in latency if r.get('teacher') == domain and r['success']]
            if pts:
                ax.scatter(*zip(*pts), s=10, color=color, alpha=0.6, label=domain, edgecolors='none')
        style(ax, 'Teacher scoring latency per sample', 'seconds', 'sequence tokens')
        legend(ax)
    panel('teacher_latency', teacher_latency)

    columns = 3
    rows = -(-len(panels) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 3.6 * rows), facecolor=SURFACE)
    for ax, (name, draw) in zip(axes.flat, panels):
        draw(ax)
        single, single_ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
        draw(single_ax)
        single.tight_layout()
        single.savefig(out / f'{name}.png', dpi=130)
        plt.close(single)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    figure.suptitle(result.name, x=0.01, ha='left', fontsize=13, color=INK, fontweight='bold')
    figure.tight_layout()
    figure.savefig(out / 'dashboard.png', dpi=110)
    print(out / 'dashboard.png')


if __name__ == '__main__':
    main()
