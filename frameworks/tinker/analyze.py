"""Summarize retained Tinker runs, including invalidated and failed attempts."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import statistics


def read_jsonl(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    # A live writer may be midway through the final record.
    result = []
    for i, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1:
                raise
    return result


def summarize(path):
    config = json.loads((path/'config.json').read_text())
    events = read_jsonl(path/'events.jsonl')
    steps = read_jsonl(path/'metrics.jsonl')
    evaluations = read_jsonl(path/'evaluations.jsonl')
    counters = Counter()
    for row in read_jsonl(path/'samples.jsonl'):
        prefix = row['split']
        counters[f'{prefix}_samples'] += 1
        counters[f'{prefix}_prompt_tokens'] += len(row['prompt_tokens'])
        counters[f'{prefix}_response_tokens'] += len(row['tokens'])
    sample_prompt_tokens = counters['train_prompt_tokens'] + counters['dev_prompt_tokens']
    sample_response_tokens = counters['train_response_tokens'] + counters['dev_response_tokens']
    train_tokens = sum(s['prompt_tokens'] + s['response_tokens'] - s['samples'] for s in steps)
    teacher_tokens = sum(len(r['prompt_tokens']) + len(r['tokens']) for r in read_jsonl(path/'teacher-scores.jsonl'))
    # Public prices observed 2026-09-21; not settled billing. Counts exclude
    # unfinished mutations, unlogged retries, standalone probes, and storage.
    common_cost = (sample_response_tokens*1.335 + train_tokens*1.177)/1e6
    estimates = {'all_prefill_cached_usd': common_cost+(sample_prompt_tokens+teacher_tokens)*0.108/1e6,
        'all_prefill_uncached_usd': common_cost+(sample_prompt_tokens+teacher_tokens)*0.54/1e6,
        'training_tokens_completed_steps': train_tokens, 'teacher_scoring_tokens': teacher_tokens,
        'status': 'partial recorded-token estimate; not an invoice or full campaign cost'}
    invalid = path/'INVALIDATED.json'
    status = 'invalidated' if invalid.exists() else events[-1]['event'] if events else 'unknown'
    info_path = path/'model-info.json'
    model_info = json.loads(info_path.read_text()) if info_path.exists() else None
    return {'name': path.name, 'status': status, 'mode': config['mode'],
        'domain': config['domain'], 'rank': config['rank'], 'learning_rate': config['learning_rate'],
        'completed_updates': len(steps), 'sample_counts': dict(counters), 'token_cost_estimate': estimates,
        'elapsed_seconds': events[-1]['elapsed_seconds'] if events else None,
        'started_utc': events[0]['utc'] if events else None,
        'last_event_utc': events[-1]['utc'] if events else None,
        'sum_step_seconds': sum(s['step_seconds'] for s in steps),
        'sum_sampling_seconds': sum(s['sampling_seconds'] for s in steps),
        'sum_teacher_scoring_seconds': sum(s['teacher_seconds'] for s in steps),
        'sum_train_seconds': sum(s['train_seconds'] for s in steps),
        'sum_evaluation_seconds': sum(e['seconds'] for e in evaluations),
        'median_step_seconds': statistics.median(s['step_seconds'] for s in steps) if steps else None,
        'evaluations': evaluations, 'checkpoints': read_jsonl(path/'checkpoints.jsonl'),
        'model_info': model_info, 'path': str(path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    args = p.parse_args()
    runs = [summarize(x.parent) for x in sorted(args.root.glob('*/config.json'))]
    (args.root/'summary.json').write_text(json.dumps(runs,indent=2))
    lines = ['# Tinker MOPD execution report', '',
        'Live summary generated from retained artifacts. Invalidated attempts must not be used as benchmark results.', '',
        '| Attempt | Status | Updates | LR | Median step (s) | Elapsed (s) |',
        '|---|---|---:|---:|---:|---:|']
    for r in runs:
        median = f"{r['median_step_seconds']:.2f}" if r['median_step_seconds'] else '—'
        lines.append(f"| {r['name']} | {r['status']} | {r['completed_updates']} | {r['learning_rate']:g} | {median} | {r['elapsed_seconds']:.2f} |")
    starts=[datetime.fromisoformat(r['started_utc']) for r in runs if r['started_utc']]
    ends=[datetime.fromisoformat(r['last_event_utc']) for r in runs if r['last_event_utc']]
    if starts and ends:
        lines += ['', f"Observed campaign interval from the first recorded attempt to the latest event: {(max(ends)-min(starts)).total_seconds()/60:.2f} minutes. This includes gaps and overlapping teacher runs, and excludes research/setup before the first attempt.", '']
    students=[r for r in runs if r['mode']=='mopd']
    for r in students:
        lines += ['', f"## Student execution: {r['name']}", '',
            f"Observed interval: {r['started_utc']} to {r['last_event_utc']}. Status: **{r['status']}**.", '',
            f"Client elapsed: {r['elapsed_seconds']:.2f} s; sum of update loops: {r['sum_step_seconds']:.2f} s; evaluations: {r['sum_evaluation_seconds']:.2f} s.", '',
            f"Within the update loops, sampling took {r['sum_sampling_seconds']:.2f} s, frozen-teacher scoring {r['sum_teacher_scoring_seconds']:.2f} s, and learner requests/logging {r['sum_train_seconds']:.2f} s. Remaining loop time includes weight publication. These are client-observed wall times, not isolated GPU compute times.", '']
    selection_path=args.root/'teacher-selection.json'
    if selection_path.exists():
        selection=json.loads(selection_path.read_text())
        lines += ['', '## Selected frozen teachers', '',
            '| Domain | Selected update | Own-domain dev accuracy | Provisional goal | Goal met |',
            '|---|---:|---:|---:|---|']
        for domain, s in selection['selected'].items():
            lines.append(f"| {domain} | {s['step']} | {s['evaluation'][domain]['accuracy']:.2%} | {s['provisional_target']:.0%} | {s['target_met']} |")
        lines += ['', 'These goals are operational surrogates, not the missing original teacher scores. The teacher-selection JSON retains exact private remote checkpoint paths and evaluations. The student is initialized independently from the base model, not from either teacher.', '']
    lines += ['', '## Development evaluation', '',
        'All scores below use 512 examples per domain unless the sample count says otherwise. Temperature 0, response cap 256, thinking disabled.', '',
        '| Attempt | Update | Countdown | Graph Coloring | N/domain |',
        '|---|---:|---:|---:|---:|']
    for r in runs:
        for e in r['evaluations']:
            lines.append(f"| {r['name']} | {e['step']} | {e['countdown']['accuracy']:.2%} | {e['graph_color']['accuracy']:.2%} | {e['countdown']['n']}/{e['graph_color']['n']} |")
    lines += ['', '## Recorded-token cost estimates', '',
        'Illustrative all-cached versus all-uncached prefill cases at the [published model prices](https://tinker-docs.thinkingmachines.ai/tinker/models/): $0.108/$0.54 per million prefill tokens, $1.335 per million generated tokens, $1.177 per million training tokens. These are not invoices or bounds on total spend. They exclude unlogged retries, unfinished mutations, standalone probes, and checkpoint storage. Settled billing is authoritative.', '',
        '| Attempt | All prefill cached | All prefill uncached |', '|---|---:|---:|']
    for r in runs:
        c = r['token_cost_estimate']
        lines.append(f"| {r['name']} | ${c['all_prefill_cached_usd']:.3f} | ${c['all_prefill_uncached_usd']:.3f} |")
    cached=sum(r['token_cost_estimate']['all_prefill_cached_usd'] for r in runs)
    uncached=sum(r['token_cost_estimate']['all_prefill_uncached_usd'] for r in runs)
    lines.append(f"| Total recorded attempts | ${cached:.3f} | ${uncached:.3f} |")
    lines += ['', '## Original self-hosted observations', '',
        'From `docs/PROVENANCE.md` in the OPD repo; one run each, different objectives and scheduling. Durations include successful allocation startup/evaluation/checkpoint/cleanup, and exclude earlier failures.', '',
        '| Framework | Countdown baseline → final | Graph Coloring baseline → final | Successful allocation |',
        '|---|---:|---:|---:|',
        '| Miles | 10.55% → 45.51% | 26.17% → 88.28% | 21m 34s |',
        '| Prime-RL | 10.94% → 39.65% | 25.78% → 64.65% | 24m 22s |',
        '| Slime | 10.94% → 44.14% | 26.37% → 89.26% | 13m 25s |', '',
        'The exact original teacher development scores are missing from the supplied sources; teacher-quality matching is not established. Tinker is rank-64 LoRA with a documented 10× LR adjustment, remote provider-controlled weights/hardware, synchronous rollouts, and sampled-token distillation. This is a workload-matched service observation, not an isolated framework ranking.', '',
        'Teacher runs may overlap in wall time. Report their individual elapsed times and total campaign elapsed time separately; do not sum overlapping durations as elapsed wall time. Hosted cost must include invalidated attempts and specialist training, with consolidation cost reported separately. Billing data can lag by hours.', '',
        'See `JOURNAL.md`, `protocol.md`, per-run JSONL, console logs, and remote checkpoint records for evidence. No model weights were downloaded.', '']
    audit_path=args.root/'final-audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        lines += ['## Final artifact audit', '',
            f"Passed: **{audit['passed']}**. Verified {audit['optimizer_updates']} student updates, {audit['training_sequences']} training sequences and {audit['development_sequences']} development sequences; recomputed all task scores and teacher-minus-student advantages. See `final-audit.json` and `frameworks/tinker/audit.py`.", '']
    metadata_dirs=sorted(args.root.glob('service-metadata-*'))
    if metadata_dirs:
        metadata=metadata_dirs[-1]
        checkpoints={}
        for path in metadata.glob('*-checkpoints.json'):
            for checkpoint in json.loads(path.read_text()).get('checkpoints',[]):
                checkpoints[checkpoint['tinker_path']]=checkpoint
        size=sum(c['size_bytes'] for c in checkpoints.values())
        lines += ['## Remote retention and billing', '',
            f"Metadata snapshot: `{metadata.name}`. It lists {len(checkpoints)} persistent remote checkpoints totaling {size:,} bytes ({size/1e9:.2f} decimal GB). No checkpoint files were downloaded. At the documented $0.10/GB-month storage rate, retaining this amount for a full month would be approximately ${size/1e9*.1:.2f}, subject to provider billing units and retention time. Storage is separate from token usage.", '']
        billing_path=metadata/'billing.json'
        if billing_path.exists():
            billing=json.loads(billing_path.read_text())
            if billing.get('data')==[]:
                lines += ['The billing snapshot contains no settled usage rows. This does **not** mean zero spend; billing can lag. Re-query the retained billing command later for the invoice evidence.', '']
            else:
                lines += ['Billing rows are retained in the snapshot. Use their settled provider amounts, not the illustrative token estimates above, for a cost claim.', '']
    (args.root/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps([{k:r[k] for k in ('name','status','completed_updates','sample_counts')} for r in runs],indent=2))


if __name__ == '__main__':
    main()
