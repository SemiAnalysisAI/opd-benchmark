"""Render the report's figures from `data/` (benchmark summaries, telemetry.json, gpu-series.csv).

    uv run --with matplotlib python docs/caesar-opd-2026-09-27/build_charts.py

Each framework keeps one color in every figure; base and teacher are grey. Two of the colors
fall below 3:1 contrast on the light background, so every series is also labeled directly and
every value appears in the report's tables.
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / 'data', HERE / 'figures'
SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1'
FRAMEWORKS = {  # Fixed order; color follows the framework, not its rank.
    'miles-1021': ('Miles', '#2a78d6'), 'prime-rl-1017': ('Prime-RL', '#eb6834'),
    'slime-979': ('Slime', '#1baf7a'), 'verl-966': ('verl', '#eda100')}
REFERENCES = {'base': ('Base', '#b7b6b1'), 'teacher-caesar': ('Caesar teacher', '#6f6e69')}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlelocation': 'left',
    'axes.titlesize': 11, 'axes.titlecolor': INK, 'lines.linewidth': 2, 'lines.solid_capstyle': 'round'})


def end_labels(ax, labels, min_gap=0.06):
    """Label each line at its end, nudging labels apart vertically so they do not overlap."""
    low, high = ax.get_ylim()
    span = high - low
    placed = []
    for x, y, text in sorted(labels, key=lambda item: item[1]):
        y = max(y, placed[-1] + min_gap * span) if placed else y
        placed.append(y)
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords='offset points', va='center', color=INK_2, fontsize=8.5)


def available(names):
    return [n for n in names if (DATA / n / 'summary.json').exists()]


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=180, bbox_inches='tight')
    plt.close(fig)


def quality():
    """Final-model avg@8 on both dev sets, one panel per task on the same 0-100 scale."""
    models = available([*REFERENCES, *FRAMEWORKS])
    summaries = {m: json.loads((DATA / m / 'summary.json').read_text()) for m in models}
    style = {**REFERENCES, **FRAMEWORKS}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (domain, title) in zip(axes, (('caesar_cipher', 'caesar_cipher (trained task)'),
                                          ('simple_geometry', 'simple_geometry (not trained)'))):
        values = [100 * summaries[m]['domains'][domain]['avg@8'] for m in models]
        bars = ax.bar(range(len(models)), values, width=0.62, color=[style[m][1] for m in models],
                      edgecolor=SURFACE, linewidth=2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f'{value:.1f}', ha='center', va='bottom',
                    color=INK, fontsize=9)
        ax.set_xticks(range(len(models)), [style[m][0] for m in models], rotation=20, ha='right')
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('avg@8 (%)')
    fig.suptitle('Final-model accuracy, dev set, 8 samples per prompt, temperature 1, thinking on',
                 x=0.01, ha='left', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, 'quality.png')


def updates():
    runs = {r['run']: r for r in json.loads((DATA / 'telemetry.json').read_text()) if r.get('updates')}
    return {run: runs[run]['updates']['per_update'] for run in FRAMEWORKS if run in runs}


def per_update_panels():
    """One panel per per-update metric, all frameworks overlaid."""
    series = updates()
    panels = [('step_s', 'Update time (s)', 1),
              ('wait_share', 'Share of update waiting for rollouts (%)', 100),
              ('response_tokens_mean', 'Mean trained response (k tokens)', 1e-3),
              ('reverse_kl', 'Reverse KL to teacher (sampled tokens)', 1)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
    for ax, (field, title, scale) in zip(axes.flat, panels):
        labels = []
        for run, rows in series.items():
            name, color = FRAMEWORKS[run]
            xs, ys = [], []
            for i, row in enumerate(rows, start=1):
                value = (row['wait_for_rollouts_s'] / row['step_s']
                         if field == 'wait_share' and row['wait_for_rollouts_s'] and row['step_s'] else row.get(field))
                if isinstance(value, (int, float)):
                    xs.append(i if run != 'verl-966' else i + 1)  # verl logs no line for its first update.
                    ys.append(value * scale)
            ax.plot(xs, ys, color=color, marker='o', markersize=3.5)
            labels.append((xs[-1], ys[-1], name))
        ax.set_title(title)
        ax.set_xlim(0.5, 23)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel('Update')
        end_labels(ax, labels)
    fig.suptitle('Per-update training metrics from each framework\'s own logs (updates 1 and 11 include in-loop evals)',
                 x=0.01, ha='left', color=INK, fontsize=12)
    fig.legend(handles=[plt.Line2D([], [], color=FRAMEWORKS[r][1], marker='o', label=FRAMEWORKS[r][0])
                        for r in series], loc='upper left', bbox_to_anchor=(0.01, 0.955), ncol=len(series),
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, 'per_update.png')


# Capture roles follow the site file, but Ray put verl's rollout on the "learner" node.
ROLLOUT_NODE_ROLE = {'verl-966': 'learner'}


def rolling(values, width=12):
    """Trailing mean over `width` samples (12 x 5 s = 1 minute)."""
    return [sum(values[max(0, i - width + 1):i + 1]) / len(values[max(0, i - width + 1):i + 1])
            for i in range(len(values))]


def gpu_utilization():
    """Mean utilization of each node's used GPUs over the run, one panel per framework."""
    points = {}
    with (DATA / 'gpu-series.csv').open() as stream:
        for row in csv.DictReader(stream):
            points.setdefault(row['run'], {}).setdefault(row['role'], []).append(
                (float(row['seconds']) / 60, float(row['gpu_util_mean'])))
    runs = [r for r in FRAMEWORKS if r in points]
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 1.9 * len(runs)), sharex=True, sharey=True)
    for ax, run in zip(axes, runs):
        name, color = FRAMEWORKS[run]
        rollout_role = ROLLOUT_NODE_ROLE.get(run, 'generation')
        for role in sorted(points[run], key=lambda r: r != rollout_role):  # Rollout node first.
            xs, ys = zip(*points[run][role])
            rollout = role == rollout_role
            ax.plot(xs, rolling(list(ys)), color=color if rollout else MUTED, linewidth=1.6,
                    label='rollout + teacher node' if rollout else 'trainer node')
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 105)
        ax.set_ylabel('GPU util (%)')
        ax.legend(loc='lower right', bbox_to_anchor=(1, 1.0), frameon=False, fontsize=8, ncol=2)
    axes[-1].set_xlabel('Minutes since the allocation started')
    fig.suptitle('GPU utilization per node over the whole run (mean of the GPUs the run used; 1-minute rolling mean)',
                 x=0.01, ha='left', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, 'gpu_utilization.png')


if __name__ == '__main__':
    quality()
    per_update_panels()
    gpu_utilization()
    print('figures written to', OUT)
