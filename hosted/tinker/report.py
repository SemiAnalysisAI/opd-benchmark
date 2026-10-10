"""Audit the MOPD student and summarize every run of a Tinker campaign.

Writes final-audit.json, summary.json, REPORT.md, and, when matplotlib is
installed, learning-curves.png. Only the retained run records are read.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared import recipe  # noqa: E402
from shared.puzzles import load_split, manifest  # noqa: E402

# The fixed student protocol and the record counts it implies.
DOMAINS = recipe.DOMAINS
UPDATES, PROMPTS, EVAL_SIZE = recipe.UPDATES, recipe.PROMPTS_PER_UPDATE, recipe.EVAL_EXAMPLES  # Full dev splits.
EVAL_STEPS = sorted({*range(0, UPDATES + 1, recipe.EVAL_INTERVAL), UPDATES})
TRAIN_SEQUENCES = UPDATES * PROMPTS
DEV_SEQUENCES = len(EVAL_STEPS) * sum(EVAL_SIZE.values())
# Published Tinker prices observed 2026-09-21, USD per million tokens. Illustrative, not billing.
PRICE = {'generated': 1.335, 'training': 1.177, 'prefill_cached': 0.108, 'prefill_uncached': 0.54}


def read_jsonl(path):
    """Rows of a JSONL file. A live writer may leave only the final line incomplete."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return rows


def key(r):
    return (r['step'], r['domain'], r['index'], r['sample'])


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sampled(r):
    """The sampled fields that the teacher-score records must repeat unchanged."""
    return digest({k: r[k] for k in ('prompt_tokens', 'tokens', 'logprobs', 'response', 'score')})


def audit(root):
    """Check that the student ran the fixed protocol and that its records are consistent."""
    run = root/'student'
    config = json.loads((run/'config.json').read_text())
    assert config['mode'] == 'mopd' and config['checkpoint'] is None and config['start_step'] == 0
    assert (config['steps'], config['prompts'], config['group_size'], config['max_tokens'], config['eval_every']) \
        == (UPDATES, PROMPTS, 1, recipe.MAX_RESPONSE_TOKENS, recipe.EVAL_INTERVAL)
    pipelined = config.get('pipeline_depth', 0) == 1
    events = read_jsonl(run/'events.jsonl')
    assert events[-1]['event'] == 'complete' and events[-1]['optimizer_updates'] == UPDATES
    metrics = read_jsonl(run/'metrics.jsonl')
    assert [m['step'] for m in metrics] == list(range(1, UPDATES + 1))
    for m in metrics:
        assert m['samples'] == PROMPTS and 0 <= m['accepted_policy_lag'] <= int(pipelined) * recipe.MAX_POLICY_LAG
        if pipelined:
            p = m['pipeline']
            assert p['learner_version'] == m['step'] - 1 and p['policy_version'] == max(0, m['step'] - 2)
            assert m['accepted_policy_lag'] == p['learner_version'] - p['policy_version']
            assert p['discarded_responses'] == 0

    # Rescore every sample with the run's retained scorer on hash-checked data.
    recorded = json.loads((run/'data-manifest.json').read_text())
    dataset = {}
    for d in DOMAINS:
        for split in ('train', 'dev'):
            name = recipe.data_file(d, split)
            assert manifest()[name]['sha256'] == recorded[name]['sha256']
            dataset[d, split] = load_split(d, split)
    spec = importlib.util.spec_from_file_location('retained_scorer', run/'scoring.py')
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    counts, correct, train, dev = Counter(), Counter(), {}, set()
    for r in read_jsonl(run/'samples.jsonl'):
        assert scorer.score(r['response'], dataset[r['domain'], r['split']][r['index']]['label']) == r['score']
        assert 0 < len(r['tokens']) <= recipe.MAX_RESPONSE_TOKENS and len(r['tokens']) == len(r['logprobs'])
        assert all(math.isfinite(x) for x in r['logprobs']) and r['sample'] == 0
        counts[r['split'], r['step'], r['domain']] += 1
        correct[r['split'], r['step'], r['domain']] += r['score']
        if r['split'] == 'train':
            assert key(r) not in train and (not pipelined or r['policy_version'] == max(0, r['step'] - 1))
            train[key(r)] = {'sampled': sampled(r), 'length': len(r['tokens'])}
        else:
            assert key(r) not in dev
            dev.add(key(r))
    assert len(train) == TRAIN_SEQUENCES and sum(counts.values()) == TRAIN_SEQUENCES + DEV_SEQUENCES
    assert all(counts['train', step, d] == PROMPTS // len(DOMAINS) for step in range(UPDATES) for d in DOMAINS)
    assert all(counts['dev', step, d] == EVAL_SIZE[d] for step in EVAL_STEPS for d in DOMAINS)
    assert dev == {(step, d, i, 0) for step in EVAL_STEPS for d in DOMAINS for i in range(EVAL_SIZE[d])}
    evaluations = read_jsonl(run/'evaluations.jsonl')
    assert [e['step'] for e in evaluations] == EVAL_STEPS
    for e in evaluations:
        for d in DOMAINS:
            assert e[d]['correct'] == correct['dev', e['step'], d]
            assert e[d]['accuracy'] == correct['dev', e['step'], d] / EVAL_SIZE[d]

    # Each training sample has one teacher score; its advantage is teacher minus sampler logprob.
    advantages, gap_sums, gap_counts = {}, defaultdict(float), Counter()
    for r in read_jsonl(run/'teacher-scores.jsonl'):
        assert not pipelined or r['policy_version'] == max(0, r['step'] - 1)
        k = key(r)
        assert k in train and k not in advantages and sampled(r) == train[k]['sampled']
        advantages[k] = json.dumps(r['advantages'])
        assert len(r['tokens']) == len(r['teacher_logprobs']) == len(r['advantages'])
        for t, s, a in zip(r['teacher_logprobs'], r['logprobs'], r['advantages']):
            assert math.isfinite(a) and math.isclose(a, t - s, rel_tol=0, abs_tol=1e-12)
            gap_sums[r['step'], r['domain']] -= a
            gap_counts[r['step'], r['domain']] += 1
    assert advantages.keys() == train.keys()
    for m in metrics:
        for d in DOMAINS:
            k = (m['step'] - 1, d)
            assert math.isclose(m['sampled_reverse_kl_by_domain'][d], gap_sums[k] / gap_counts[k], abs_tol=1e-10)

    # The learner trained on every sample once, with the teacher's advantages.
    learner = read_jsonl(run/'learner-scores.jsonl')
    assert len(learner) == TRAIN_SEQUENCES and {key(r) for r in learner} == train.keys()
    for r in learner:
        assert len(r['logprobs']) == len(r['advantages']) == train[key(r)]['length']
        assert all(math.isfinite(x) for x in r['logprobs']) and json.dumps(r['advantages']) == advantages[key(r)]

    # The student used the selected teachers and is not one of them.
    routes = json.loads((run/'teachers.json').read_text())
    selection = json.loads((root/'teacher-selection.json').read_text())
    assert routes == {d: v['checkpoint']['sampler_path'] for d, v in selection['selected'].items()}
    student_id = json.loads((run/'model-info.json').read_text())['model_id']
    assert all(student_id not in path for path in routes.values())
    return {'passed': True, 'optimizer_updates': UPDATES, 'training_sequences': TRAIN_SEQUENCES,
        'training_sequences_per_domain': TRAIN_SEQUENCES // len(DOMAINS), 'development_sequences': DEV_SEQUENCES,
        'exact_data_hashes': True, 'all_sample_scores_recomputed': True,
        'exact_teacher_minus_student_advantages': True, 'balanced_routing_each_update': True,
        'learner_and_teacher_scores_cover_all_training_sequences': True,
        'fresh_student': True, 'zero_accepted_policy_lag': all(m['accepted_policy_lag'] == 0 for m in metrics),
        'max_accepted_policy_lag': max(m['accepted_policy_lag'] for m in metrics),
        'pipeline_depth': int(pipelined),
        'source_sha256': hashlib.sha256((run/'source.py').read_bytes()).hexdigest()}


def summarize(path):
    """Status, stage timing, evaluations, and recorded-token cost of one run directory."""
    config = json.loads((path/'config.json').read_text())
    events = read_jsonl(path/'events.jsonl')
    steps = read_jsonl(path/'metrics.jsonl')
    evaluations = read_jsonl(path/'evaluations.jsonl')
    tokens = Counter()
    for r in read_jsonl(path/'samples.jsonl'):
        tokens['prompt'] += len(r['prompt_tokens'])
        tokens['generated'] += len(r['tokens'])
    tokens['training'] = sum(s['prompt_tokens'] + s['response_tokens'] - s['samples'] for s in steps)
    tokens['teacher'] = sum(len(r['prompt_tokens']) + len(r['tokens']) for r in read_jsonl(path/'teacher-scores.jsonl'))
    common = (tokens['generated']*PRICE['generated'] + tokens['training']*PRICE['training'])/1e6
    prefill = tokens['prompt'] + tokens['teacher']
    # Fractions of the update loops; the residual is weight publication and other client work.
    stages = {k: sum(s[k] for s in steps) for k in ('sampling_seconds', 'teacher_seconds', 'train_seconds')}
    stages['publication_and_other_seconds'] = sum(s['step_seconds'] for s in steps) - sum(stages.values())
    status = 'invalidated' if (path/'INVALIDATED.json').exists() else events[-1]['event'] if events else 'unknown'
    return {'name': path.name, 'status': status, 'mode': config['mode'], 'domain': config['domain'],
        'learning_rate': config['learning_rate'], 'pipeline_depth': config.get('pipeline_depth', 0),
        'completed_updates': len(steps), 'tokens': dict(tokens),
        'elapsed_seconds': events[-1]['elapsed_seconds'] if events else None,
        'median_step_seconds': statistics.median(s['step_seconds'] for s in steps) if steps else None,
        'stage_seconds': stages,
        'evaluation_seconds': sum(e['seconds'] for e in evaluations), 'evaluations': evaluations,
        'cost_all_prefill_cached_usd': common + prefill*PRICE['prefill_cached']/1e6,
        'cost_all_prefill_uncached_usd': common + prefill*PRICE['prefill_uncached']/1e6}


def plot(root, runs):
    """Teacher and student development accuracy per domain."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed; skipping learning-curves.png')
        return
    baseline = next((r['evaluations'][0] for r in runs if r['mode'] == 'eval' and r['status'] == 'complete'), None)
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(11, 4.6), sharey=True)
    for ax, d in zip(axes, DOMAINS):
        teacher = {0: baseline[d]['accuracy']*100} if baseline else {}
        for r in runs:
            if r['mode'] == 'teacher' and r['domain'] == d and r['status'] != 'invalidated':
                teacher.update({e['step']: e[d]['accuracy']*100 for e in r['evaluations']})
        ax.plot(sorted(teacher), [teacher[x] for x in sorted(teacher)], '-o', color='#156f63', label='Specialist teacher')
        for r in runs:
            if r['mode'] == 'mopd' and r['status'] != 'invalidated' and r['evaluations']:
                es = r['evaluations']
                ax.plot([e['step'] for e in es], [e[d]['accuracy']*100 for e in es], '-o', color='#a047a5', label=r['name'])
        ax.set(title=d, xlabel='Optimizer updates', ylim=(0, 100))
        ax.legend(frameon=False, loc='lower right')
    axes[0].set_ylabel('Development accuracy (%)')
    fig.suptitle('Tinker: routed MOPD student (full dev split, greedy, thinking on)')
    fig.tight_layout()
    fig.savefig(root/'learning-curves.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def report(root, runs, final_audit):
    lines = ['# Tinker MOPD report', '',
        '| Run | Status | Updates | LR | Median update (s) | Sampling | Teacher | Learner | Other | Evaluation (s) | Elapsed (s) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        total = sum(r['stage_seconds'].values())
        fractions = [f'{v/total:.0%}' if total else '—' for v in r['stage_seconds'].values()]
        median = f"{r['median_step_seconds']:.2f}" if r['median_step_seconds'] else '—'
        elapsed = f"{r['elapsed_seconds']:.0f}" if r['elapsed_seconds'] is not None else '—'
        lines.append(f"| {r['name']} | {r['status']} | {r['completed_updates']} | {r['learning_rate']:g} | {median} | "
                     + ' | '.join(fractions) + f" | {r['evaluation_seconds']:.0f} | {elapsed} |")
    selection_path = root/'teacher-selection.json'
    if selection_path.exists():
        lines += ['', '## Selected teachers', '', '| Domain | Run | Update | Own-domain dev accuracy | Goal | Met |',
                  '|---|---|---:|---:|---:|---|']
        for d, s in json.loads(selection_path.read_text())['selected'].items():
            lines.append(f"| {d} | {s['source_run']} | {s['step']} | {s['evaluation'][d]['accuracy']:.2%} "
                         f"| {s['provisional_target']:.0%} | {s['target_met']} |")
    lines += ['', '## Development accuracy', '', '| Run | Update | ' + ' | '.join(DOMAINS) + ' |',
              '|---|---:|' + '---:|' * len(DOMAINS)]
    for r in runs:
        for e in r['evaluations']:
            lines.append(f"| {r['name']} | {e['step']} | "
                         + ' | '.join(f"{e[d]['correct']:g}/{e[d]['n']} ({e[d]['accuracy']:.1%})" for d in DOMAINS) + ' |')
    lines += ['', '## Recorded-token cost estimate (USD)', '', '| Run | All prefill cached | All prefill uncached |', '|---|---:|---:|']
    lines += [f"| {r['name']} | {r['cost_all_prefill_cached_usd']:.3f} | {r['cost_all_prefill_uncached_usd']:.3f} |" for r in runs]
    lines.append(f"| Total | {sum(r['cost_all_prefill_cached_usd'] for r in runs):.3f} "
                 f"| {sum(r['cost_all_prefill_uncached_usd'] for r in runs):.3f} |")
    if final_audit:
        lines += ['', f"Audit passed: {final_audit['optimizer_updates']} student updates, {final_audit['training_sequences']} "
                  f"training and {final_audit['development_sequences']} development sequences rescored; "
                  f"maximum accepted policy lag {final_audit['max_accepted_policy_lag']}. See final-audit.json."]
    prices = ', '.join(f'{k} ${v:g}' for k, v in PRICE.items())
    lines += ['', '## Caveats', '',
        '- Development-split scores from one seed, greedy, thinking on. Teacher goals are operational (just below the '
        'released GRPO teachers\' held-out scores); the retrained teachers are not the frozen recipe teachers.',
        '- Stage shares are client wall time within update loops (network and queueing included), not GPU utilization. '
        'In pipelined runs, Sampling is the exposed wait for the prepared batch; teacher scoring overlaps it in the background.',
        f'- Costs use recorded tokens at published per-million prices ({prices}); they are not invoices and exclude retries and storage.',
        '- Tinker trains rank-64 LoRA with a 10x learning rate on provider-controlled hardware: a workload-matched observation, '
        'not a controlled framework ranking.', '']
    (root/'REPORT.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    root = p.parse_args().root.resolve()
    final_audit = None
    if (root/'student').exists():
        final_audit = audit(root)
        (root/'final-audit.json').write_text(json.dumps(final_audit, indent=2))
    runs = [summarize(path.parent) for path in sorted(root.glob('*/config.json'))]
    (root/'summary.json').write_text(json.dumps(runs, indent=2))
    report(root, runs, final_audit)
    plot(root, runs)
    print(json.dumps({'audit_passed': bool(final_audit), 'runs': [
        {k: r[k] for k in ('name', 'status', 'completed_updates')} for r in runs]}, indent=2))


if __name__ == '__main__':
    main()
