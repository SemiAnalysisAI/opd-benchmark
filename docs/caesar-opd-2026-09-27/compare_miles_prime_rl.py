"""Miles (965) against Prime-RL on upstream defaults (1017), with Prime-RL's first run (968) as a dashed reference.

Usage: `uv run --with matplotlib python docs/caesar-opd-2026-09-27/compare_miles_prime_rl.py`; writes
`figures/miles_vs_prime_rl.png` from `data/`.
"""
import json

import matplotlib.pyplot as plt

from build_charts import DATA, INK, INK_2, MUTED, SURFACE, end_labels, save

MILES, PRIME, PRIME_OLD = 'miles-965', 'prime-rl-1017', 'prime-rl-968'
STYLE = {  # Colors match the report's figures; the first run is a muted, dashed reference.
    MILES: dict(label='Miles', color='#2a78d6', linestyle='-'),
    PRIME: dict(label='Prime-RL (defaults, 1017)', color='#eb6834', linestyle='-'),
    PRIME_OLD: dict(label='Prime-RL (first run, 968)', color='#b7b6b1', linestyle='--'),
}
TEACHER = 'teacher-caesar'
# Policy-engine generation throughput over the whole run, from the report's GPU table (engine counters).
POLICY_TOKENS_PER_S = {MILES: 18.3, PRIME: 40.3, PRIME_OLD: 20.9}


def telemetry():
    runs = {r['run']: r for r in json.loads((DATA / 'telemetry.json').read_text())}
    runs.update({r['run']: r for r in json.loads((DATA / 'telemetry-prime-rl-968.json').read_text())})
    return {run: runs[run]['updates']['per_update'] for run in STYLE}


def bars(ax, runs, values, fmt, reference=None):
    xs = range(len(runs))
    drawn = ax.bar(xs, values, width=0.6, color=[STYLE[r]['color'] for r in runs], edgecolor=SURFACE, linewidth=2)
    for bar, run in zip(drawn, runs):
        if STYLE[run]['linestyle'] == '--':
            bar.set_hatch('///')
            bar.set_edgecolor(SURFACE)
    top = max(values + ([reference] if reference else []))
    for x, value in zip(xs, values):
        ax.text(x, value + 0.02 * top, fmt.format(value), ha='center', va='bottom', color=INK, fontsize=9)
    if reference:  # Named in the panel title, so the line needs no label of its own.
        ax.axhline(reference, color=MUTED, linewidth=1, linestyle=':')
    ax.set_xticks(list(xs), [STYLE[r]['label'].replace(' (', '\n(') for r in runs], fontsize=8.5)
    ax.set_ylim(0, top * 1.18)
    ax.grid(axis='x', visible=False)


def lines(ax, series, field, scale=1.0, cumulative=False):
    labels = []
    for run, rows in series.items():
        ys, total = [], 0.0
        for row in rows:
            value = row[field] * scale
            total += value
            ys.append(total if cumulative else value)
        xs = list(range(1, len(ys) + 1))
        style = STYLE[run]
        ax.plot(xs, ys, color=style['color'], linestyle=style['linestyle'], marker='o', markersize=3.5,
                linewidth=2 if style['linestyle'] == '-' else 1.6)
        labels.append((xs[-1], ys[-1], style['label'].split(' (')[0] + (' 968' if run == PRIME_OLD else '')))
    ax.set_xlim(0.5, 23.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel('Update')
    end_labels(ax, labels)


def main():
    summaries = {m: json.loads((DATA / m / 'summary.json').read_text()) for m in (*STYLE, TEACHER)}
    series = telemetry()
    order = [MILES, PRIME, PRIME_OLD]
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    for ax, (domain, title) in zip(axes[0], (('caesar_cipher', 'caesar_cipher avg@8, trained task (%)'),
                                             ('simple_geometry', 'simple_geometry avg@8, not trained (%)'))):
        values = [100 * summaries[r]['domains'][domain]['avg@8'] for r in order]
        teacher = 100 * summaries[TEACHER]['domains'][domain]['avg@8']
        bars(ax, order, values, '{:.1f}', reference=teacher)
        ax.set_title(f'{title}\ndotted: caesar teacher {teacher:.1f}', fontsize=10.5)
        ax.set_ylim(0, 100)

    axes[1, 0].set_title('Cumulative training time (min)')
    lines(axes[1, 0], series, 'step_s', scale=1 / 60, cumulative=True)
    axes[1, 1].set_title('Time per update (s)')
    lines(axes[1, 1], series, 'step_s')

    axes[2, 0].set_title('Mean trained response (k tokens)')
    lines(axes[2, 0], series, 'response_tokens_mean', scale=1e-3)
    axes[2, 1].set_title('Policy generation throughput, whole run (k tokens/s)')
    bars(axes[2, 1], order, [POLICY_TOKENS_PER_S[r] for r in order], '{:.1f}')

    fig.suptitle('Miles vs Prime-RL: single-teacher OPD on caesar_cipher, same recipe, 20 updates, 2 x 8 B200',
                 x=0.01, ha='left', color=INK, fontsize=12)
    fig.legend(handles=[plt.Line2D([], [], color=STYLE[r]['color'], linestyle=STYLE[r]['linestyle'], marker='o',
                                   label=STYLE[r]['label']) for r in order],
               loc='upper left', bbox_to_anchor=(0.01, 0.97), ncol=3, frameon=False, fontsize=9)
    fig.text(0.01, 0.005, "Update 1 includes each run's start-up; Miles' update 11 its second in-loop eval (Prime-RL's evals overlap training). "
             'Whole allocations: Miles 63.6 min, Prime-RL 39.6 min (968: 51.3 min).\n'
             'Benchmark: 126 dev prompts x 8 samples, temperature 1, thinking on.', color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    save(fig, 'miles_vs_prime_rl.png')


if __name__ == '__main__':
    main()
