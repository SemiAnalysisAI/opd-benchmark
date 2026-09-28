"""Read-only final audit of a run root: `python report.py RUN_ROOT` writes `audit.json` and `REPORT.md`."""
from pathlib import Path
import argparse
import json
import math

from lifecycle import read_json  # Also makes `shared` importable.
from shared.recipe import DOMAINS, PROMPTS_PER_UPDATE, UPDATES

LIMITATIONS = ['Qwen3.5 base differs from the Qwen3.6 base used elsewhere.',
               'Teachers are retrained here; their targets are operational goals, not the frozen recipe teachers.',
               'FP8 rollout inference versus trainer numerical precision; no router replay.',
               'Client wall times and resource allocation are not measured hardware utilization.',
               'Conservative allocation cost estimate is not an invoice or provider billing cap.']
SWAP = ['student_state_saved', *['teacher_state_loaded'] * len(DOMAINS), 'student_state_restored']


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def stage(root, directory):
    config = read_json(directory / 'config.json')
    metrics = rows(directory / 'metrics.jsonl')
    evaluations = rows(directory / 'evaluations.jsonl')
    train = [s for s in rows(directory / 'samples.jsonl') if s['split'] == 'train']
    domains = DOMAINS if directory.name == 'student' else (directory.name[len('teacher-'):],)
    per_step = config['prompts'] * config['group_size'] // len(domains)
    first = config.get('start_step', 0) + 1
    item = {'completed_updates': len(metrics), 'logged_training_responses': len(train),
            'last_evaluation': evaluations[-1] if evaluations else None, 'checks': {
                'unique_samples': len({(s['step'], s['domain'], s['index'], s['sample']) for s in train}) == len(train),
                'aligned_finite_logprobs': all(s['tokens'] and len(s['tokens']) == len(s['logprobs'])
                                               and all(map(math.isfinite, s['logprobs'])) for s in train),
                'sequential_updates': [m['step'] for m in metrics] == list(range(first, first + len(metrics))),
                'responses_per_update': all(sum(s['step'] == m['step'] - 1 and s['domain'] == d for s in train)
                                            == per_step for m in metrics for d in domains)}}
    if (directory / 'complete.json').exists():
        item['completion'] = read_json(directory / 'complete.json')
    if directory.name == 'student':
        events = [e for e in rows(root / 'events.jsonl') if e['event'] in SWAP]
        swaps = [[e for e in events if e['step'] == m['step'] - 1] for m in metrics]
        counts = {d: sum(s['domain'] == d for s in train) for d in DOMAINS}
        item['checks'].update(
            checkpoint_swaps=all([e['event'] for e in swap] == SWAP and swap[0]['path'] == swap[-1]['path']
                                 and [e['domain'] for e in swap[1:-1]] == list(DOMAINS) for swap in swaps),
            protocol_complete=(len(metrics) == UPDATES and len(train) == UPDATES * PROMPTS_PER_UPDATE
                               and set(counts.values()) == {UPDATES * PROMPTS_PER_UPDATE // len(DOMAINS)}
                               and all(m['accepted_policy_lag'] == 0 for m in metrics)))
        item['domain_counts'] = counts
    return item


def report(root):
    root = Path(root)
    result = {'status': read_json(root / 'status.json'), 'limitations': LIMITATIONS, 'stages': {
        name: stage(root, root / name) for name in [*('teacher-' + d for d in DOMAINS), 'student'] if (root / name).exists()}}
    if (root / 'teacher-selection.json').exists():
        result['teachers'] = {d: {'step': s['step'], 'accuracy': s['evaluation'][d]['accuracy'], 'target': s['target'],
                                  'target_reached': s['evaluation'][d]['accuracy'] >= s['target']}
                              for d, s in read_json(root / 'teacher-selection.json').items()}
    result['cleanup_verified'] = (root / 'cleanup.json').exists() and read_json(root / 'cleanup.json')['verified_released']
    (root / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Fireworks hosted RL and MOPD audit', '', f"Status: {result['status']['stage']}", '']
    for name, item in result['stages'].items():
        lines.append(f"{name}: {item['completed_updates']} completed updates; "
                     f"{item['logged_training_responses']} logged training responses.")
        if item.get('completion', {}).get('reused'):
            lines.append(f"Reused trained teacher from {item['completion']['source_root']}, "
                         f"logical update {item['completion']['updates']}. No new updates in this run.")
        if item['last_evaluation']:
            last = item['last_evaluation']
            scores = '; '.join(f"{d} {last[d]['correct']:g}/{last[d]['n']} ({last[d]['accuracy']:.2%})" for d in DOMAINS)
            lines.append(f"Last evaluation at update {last['step']}: {scores}.")
    (root / 'REPORT.md').write_text('\n\n'.join([*lines, '', *LIMITATIONS]) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    report(parser.parse_args().root)
