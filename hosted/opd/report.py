"""Summarize OPD and MOPD runs: quality, throughput, time breakdown, and a cost estimate.

    python hosted/opd/report.py RUN_DIR [RUN_DIR ...] --output REPORT_DIR \\
        [--teachers-campaign hosted/tinker SFT campaign dir]

Writes REPORT.md (every chart's numbers are also a table there), summary.json, and PNG/SVG
figures. Only the retained run records are read; nothing is re-sampled. With
--teachers-campaign, the base model's and teachers' scores on the same benchmark are
shown alongside.
"""
import argparse
import json
from pathlib import Path
import statistics

# Published Tinker list prices for Qwen3.6-35B-A3B, USD per million tokens (observed 2026-09-27).
# Illustrative only: not an invoice, and prefill caching is ignored.
PRICE = {'sampled': 1.335, 'prefill': 0.54, 'trained': 1.177}
DOMAINS = ('caesar_cipher', 'simple_geometry')
# Reference palette (dataviz skill), light surface; validated for three slots, all pairs.
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
NEUTRAL = '#a8a7a2'  # "Other"-style remainder, not a series hue.


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def load(run):
    run = Path(run)
    config = json.loads((run/'config.json').read_text())
    events = read_jsonl(run/'events.jsonl')
    return {'dir': run, 'name': run.name, 'kind': config['kind'], 'domains': config['domains'], 'config': config,
            'metrics': read_jsonl(run/'metrics.jsonl'), 'evaluations': read_jsonl(run/'evaluations.jsonl'),
            'events': events, 'complete': bool(events) and events[-1]['event'] == 'complete',
            'label': 'OPD (' + config['domains'][0] + ')' if config['kind'] == 'opd' else 'MOPD (' + ' + '.join(config['domains']) + ')',
            'short': config['kind'].upper()}


def references(campaign):
    """Base and SFT-teacher scores on the same 100 x 3 benchmark, from a hosted/tinker SFT campaign."""
    if not campaign:
        return {}
    campaign = Path(campaign)
    base = json.loads((campaign/'baseline/benchmark.json').read_text())
    refs = {'base': {d: base[d]['accuracy'] for d in DOMAINS}, 'teacher': {}}
    for d in DOMAINS:
        teacher = json.loads((campaign/('teacher-' + d.replace('_', '-'))/'teacher.json').read_text())
        refs['teacher'][d] = teacher['benchmark']['accuracy']
    return refs


def client_seconds(metric):
    """Update time outside the service calls: saving rollouts, building the batch, computing metrics."""
    t = metric['timing']
    return t['update'] - t['rollout'] - t['train'] - t['sync']


def summarize(r):
    m, a = r['metrics'], r['config']
    timing = lambda key: [x['timing'][key] for x in m]
    tokens = {'sampled': sum(x['response_tokens'] for x in m), 'prompt': sum(x['prompt_tokens'] for x in m)}
    # Evaluation records keep mean response lengths; their prompt prefill (a few hundred tokens each) is ignored.
    eval_tokens = {'sampled': sum(e[d]['mean_tokens'] * e[d]['samples'] for e in r['evaluations'] for d in DOMAINS)}
    # Prefill: sampling prompts, teacher scoring of prompt + response. Training: prompt + response.
    prefill = tokens['prompt'] + (tokens['prompt'] + tokens['sampled'])
    trained = tokens['prompt'] + tokens['sampled']
    cost = {'sampling': (tokens['sampled'] + eval_tokens['sampled']) * PRICE['sampled'] / 1e6,
            'prefill': prefill * PRICE['prefill'] / 1e6, 'training': trained * PRICE['trained'] / 1e6}
    start, end = r['events'][0]['utc'], r['events'][-1]['utc']
    median = lambda values: statistics.median(values) if values else None
    return {
        'run': r['name'], 'kind': r['kind'], 'domains': r['domains'], 'complete': r['complete'],
        'updates': len(m), 'responses': sum(x['responses'] for x in m), 'started_utc': start, 'last_event_utc': end,
        'wall_seconds': r['events'][-1]['elapsed_seconds'],
        'update_seconds_total': sum(timing('update')), 'eval_seconds_total': sum(e['seconds'] for e in r['evaluations']),
        'median_seconds': {k: median(timing(k)) for k in ('update', 'rollout', 'train', 'sync', 'median_response',
                                                            'p90_response', 'last_response', 'teacher_tail')},
        'share_of_update_time': {**{k: sum(timing(k)) / sum(timing('update')) for k in ('rollout', 'train', 'sync')},
                                 'client': sum(client_seconds(x) for x in m) / sum(timing('update'))} if m else {},
        'median_throughput': {k: median([x['throughput'][k] for x in m]) for k in (m[0]['throughput'] if m else [])},
        'tokens': {'sampled_train': tokens['sampled'], 'prompt_train': tokens['prompt'],
                   'sampled_eval_estimate': round(eval_tokens['sampled']), 'trained': trained, 'prefill': prefill},
        'truncated_train_responses': sum(x['truncated'] for x in m),
        'train_sampler_logprob_delta_abs_mean': median([x['train_sampler_logprob_delta_abs_mean'] for x in m
                                                        if x['train_sampler_logprob_delta_abs_mean'] is not None]),
        'cost_estimate_usd': {**cost, 'total': sum(cost.values())},
        'evaluations': r['evaluations'], 'config': {k: a[k] for k in ('updates', 'prompts', 'learning_rate', 'rank',
                                                                        'max_tokens', 'eval_examples', 'eval_samples')}}


# Figures.

def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', color=INK, fontsize=11)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)


def figure_legend(fig, ax):
    """One legend in a row above the panels, clear of the data."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK, ncol=len(labels), loc='upper center',
               bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(top=0.8)


def save(fig, out, name):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0, 1, 0.9) if fig.legends else (0, 0, 1, 1))
    for ext in ('png', 'svg'):
        fig.savefig(out/f'{name}.{ext}', dpi=200, facecolor=SURFACE)


def plot_accuracy(runs, refs, out, plt):
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(10, 3.8), sharey=True)
    for ax, d in zip(axes, DOMAINS):
        style(ax, f'{d}: avg@3 on 100 dev problems', 'accuracy' if d == DOMAINS[0] else '')
        for label, value in (('SFT teacher', refs.get('teacher', {}).get(d)), ('base', refs.get('base', {}).get(d))):
            if value is not None:
                ax.axhline(value, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
                ax.annotate(f'{label} {value:.1%}', (1, value), xycoords=('axes fraction', 'data'), xytext=(0, 3),
                            textcoords='offset points', ha='right', color=MUTED, fontsize=8)
        for color, r in zip(SERIES, runs):
            points = [(e['update'], e[d]['accuracy']) for e in r['evaluations']]
            if points:
                xs, ys = zip(*points)
                trained = d in r['domains']
                ax.plot(xs, ys, color=color, linewidth=2, marker='o', markersize=5, label=r['label'],
                        linestyle='-' if trained else (0, (2, 2)))
                ax.annotate(f'{ys[-1]:.1%}', (xs[-1], ys[-1]), xytext=(6, -3), textcoords='offset points',
                            color=INK, fontsize=8)
        ax.set_xlabel('update', color=MUTED, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    axes[0].legend(frameon=False, fontsize=8, loc='lower right', labelcolor=INK)
    fig.text(0.01, 0.005, 'Dotted line: a domain the run does not train on.', color=MUTED, fontsize=7)
    save(fig, out, 'accuracy')


def plot_time(runs, out, plt):
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 3.9), sharey=True, squeeze=False)
    for ax, r in zip(axes[0], runs):
        style(ax, f'{r["short"]}: seconds per update', 'seconds' if ax is axes[0][0] else '')
        xs = [x['update'] for x in r['metrics']]
        bottom = [0.0] * len(xs)
        parts = [(SERIES[0], 'sample + teacher score', [x['timing']['rollout'] for x in r['metrics']]),
                 (SERIES[1], 'train step', [x['timing']['train'] for x in r['metrics']]),
                 (SERIES[2], 'weight sync', [x['timing']['sync'] for x in r['metrics']]),
                 (NEUTRAL, 'client-side', [client_seconds(x) for x in r['metrics']])]
        for color, name, values in parts:
            ax.bar(xs, values, bottom=bottom, color=color, width=0.8, label=name, edgecolor=SURFACE, linewidth=1)
            bottom = [b + v for b, v in zip(bottom, values)]
        ax.set_xlabel('update', color=MUTED, fontsize=9)
    figure_legend(fig, axes[0][0])
    save(fig, out, 'time-per-update')


def plot_stragglers(runs, out, plt):
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 3.9), sharey=True, squeeze=False)
    for ax, r in zip(axes[0], runs):
        style(ax, f'{r["short"]}: when responses finish', 'seconds after update start' if ax is axes[0][0] else '')
        xs = [x['update'] for x in r['metrics']]
        for color, (key, name) in zip(SERIES, (('median_response', 'median response'), ('p90_response', '90th percentile'),
                                               ('last_response', 'last response'))):
            ax.plot(xs, [x['timing'][key] for x in r['metrics']], color=color, linewidth=2, marker='o', markersize=4,
                    label=name)
        ax.set_xlabel('update', color=MUTED, fontsize=9)
    figure_legend(fig, axes[0][0])
    save(fig, out, 'response-finish-times')


def plot_kl(runs, out, plt):
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(10, 3.6), squeeze=False)
    for ax, d in zip(axes[0], DOMAINS):
        style(ax, f'{d}: sampled reverse KL to its teacher', 'nats per token' if d == DOMAINS[0] else '')
        for color, r in zip(SERIES, runs):
            if d in r['domains'] and r['metrics']:
                ax.plot([x['update'] for x in r['metrics']], [x['by_domain'][d]['reverse_kl'] for x in r['metrics']],
                        color=color, linewidth=2, marker='o', markersize=4, label=r['label'])
        ax.set_xlabel('update', color=MUTED, fontsize=9)
        if ax.lines:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    save(fig, out, 'reverse-kl')


# Markdown.

def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    return '\n'.join(lines + ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows])


def pct(value):
    return '—' if value is None else f'{value:.1%}'


def seconds(value):
    return '—' if value is None else f'{value:,.0f} s'


def markdown(runs, summaries, refs):
    lags = sorted({r['config'].get('policy_lag', 0) for r in runs})
    mode = 'synchronous (policy lag 0)' if lags == [0] else f'asynchronous lookahead (maximum policy lag {max(lags)})'
    parts = ['# OPD and MOPD on Tinker', '',
             'Fresh rank-64 LoRA students of Qwen3.6-35B-A3B, thinking on, distilled from frozen SFT teachers '
             f'(hosted/opd/sft.py). Sampled-token reverse KL, importance-sampling loss, no task reward, {mode}. '
             'Evaluation: the first 100 dev problems of each domain, 3 samples at temperature 1.', '']
    parts += ['## Quality (avg@3)', '']
    rows = []
    if refs:
        rows.append(['base model'] + [pct(refs['base'][d]) for d in DOMAINS])
        rows.append(['SFT teacher (own domain)'] + [pct(refs['teacher'][d]) for d in DOMAINS])
    for r in runs:
        for e in r['evaluations']:
            rows.append([f'{r["label"]}, update {e["update"]}'] + [pct(e[d]['accuracy']) for d in DOMAINS])
    parts += [table(['Model', *DOMAINS], rows), '', '![accuracy](accuracy.png)', '']
    parts += ['## Time and throughput', '',
              'Client wall time on a shared hosted service; not GPU utilization. Rollout is sampling with teacher '
              'scoring overlapped; the teacher tail is the scoring left after the last response finishes.', '']
    header = ['', *[s['run'] for s in summaries]]
    row = lambda name, f: [name, *[f(s) for s in summaries]]
    parts += [table(header, [
        row('complete', lambda s: 'yes' if s['complete'] else f'no ({s["updates"]} updates)'),
        row('updates × responses', lambda s: f'{s["updates"]} × {s["responses"] // max(1, s["updates"])}'),
        row('wall time', lambda s: f'{s["wall_seconds"] / 60:,.1f} min'),
        row('of which evaluation', lambda s: f'{s["eval_seconds_total"] / 60:,.1f} min'),
        row('median update', lambda s: seconds(s['median_seconds']['update'])),
        row('median rollout', lambda s: seconds(s['median_seconds']['rollout'])),
        row('median train step', lambda s: seconds(s['median_seconds']['train'])),
        row('median weight sync', lambda s: seconds(s['median_seconds']['sync'])),
        row('median response finishes', lambda s: seconds(s['median_seconds']['median_response'])),
        row('p90 response finishes', lambda s: seconds(s['median_seconds']['p90_response'])),
        row('last response finishes', lambda s: seconds(s['median_seconds']['last_response'])),
        row('teacher tail', lambda s: seconds(s['median_seconds']['teacher_tail'])),
        row('share: rollout / train / sync / client-side', lambda s: ' / '.join(f'{v:.0%}' for v in s['share_of_update_time'].values())),
        row('sampled tokens/s (median update)', lambda s: f'{s["median_throughput"].get("sampled_tokens_per_second", 0):,.0f}'),
        row('trained tokens/s (median update)', lambda s: f'{s["median_throughput"].get("trained_tokens_per_second", 0):,.0f}'),
        row('end-to-end response tokens/s', lambda s: f'{s["median_throughput"].get("update_response_tokens_per_second", 0):,.0f}'),
        row('training responses truncated', lambda s: s['truncated_train_responses']),
        row('train−sampler logprob gap (abs, median)', lambda s: f'{s["train_sampler_logprob_delta_abs_mean"]:.4f}'
            if s['train_sampler_logprob_delta_abs_mean'] is not None else '—'),
    ]), '', '![time per update](time-per-update.png)', '', '![response finish times](response-finish-times.png)', '']
    parts += ['## Tokens and estimated cost', '',
              f'List prices: sampling ${PRICE["sampled"]}/M, prefill ${PRICE["prefill"]}/M, training ${PRICE["trained"]}/M. '
              'Prefill counts the sampling prompts and the teachers scoring prompt + response; caching is ignored. '
              'An estimate, not an invoice.', '']
    parts += [table(header, [
        row('sampled tokens (training)', lambda s: f'{s["tokens"]["sampled_train"] / 1e6:,.1f} M'),
        row('sampled tokens (evaluation, est.)', lambda s: f'{s["tokens"]["sampled_eval_estimate"] / 1e6:,.1f} M'),
        row('prefill tokens', lambda s: f'{s["tokens"]["prefill"] / 1e6:,.1f} M'),
        row('trained tokens', lambda s: f'{s["tokens"]["trained"] / 1e6:,.1f} M'),
        row('estimated cost', lambda s: f'${s["cost_estimate_usd"]["total"]:,.0f}'),
    ]), '']
    parts += ['## Distillation signal', '', '![reverse KL](reverse-kl.png)', '']
    kl_rows = []
    for r in runs:
        for x in r['metrics']:
            kl_rows.append([r['label'], x['update'], *[f'{x["by_domain"][d]["reverse_kl"]:.4f}' if d in x['by_domain'] else '—'
                                                       for d in DOMAINS],
                            *[f'{x["by_domain"][d]["score"]:.0%}' if d in x['by_domain'] else '—' for d in DOMAINS]])
    parts += [table(['Run', 'update', *[f'{d} reverse KL' for d in DOMAINS], *[f'{d} train score' for d in DOMAINS]],
                    kl_rows), '']
    return '\n'.join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--teachers-campaign', type=Path)
    args = p.parse_args()
    runs = [load(r) for r in args.runs]
    refs = references(args.teachers_campaign)
    summaries = [summarize(r) for r in runs]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps({'references': refs, 'runs': summaries}, indent=2))
    (args.output/'REPORT.md').write_text(markdown(runs, summaries, refs))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed: tables only')
        return
    plt.rcParams['font.family'] = 'sans-serif'
    plot_accuracy(runs, refs, args.output, plt)
    plot_time(runs, args.output, plt)
    plot_stragglers(runs, args.output, plt)
    plot_kl(runs, args.output, plt)
    print(f'wrote {args.output}/REPORT.md')


if __name__ == '__main__':
    main()
