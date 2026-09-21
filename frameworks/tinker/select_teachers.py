"""Select completed, valid teacher checkpoints using recorded dev evaluations."""
import argparse
import json
from pathlib import Path


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def choose(rows, domain, target):
    qualifying = [r for r in rows if r['evaluation'][domain]['accuracy'] >= target]
    chosen = min(qualifying,key=lambda r:r['step']) if qualifying else max(rows,key=lambda r:(r['evaluation'][domain]['accuracy'],-r['step']))
    return {**chosen, 'provisional_target':target, 'target_met':bool(qualifying)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    targets = {'countdown':0.5, 'graph_color':0.9}
    candidates = {d:[] for d in targets}
    for config_path in sorted(root.glob('*/config.json')):
        path = config_path.parent
        config = json.loads(config_path.read_text())
        if config['mode'] != 'teacher' or (path/'INVALIDATED.json').exists():
            continue
        events = read(path/'events.jsonl')
        if events[-1]['event'] != 'complete':
            raise RuntimeError(f'Teacher attempt is not complete: {path.name}')
        checkpoints = {r['step']:r for r in read(path/'checkpoints.jsonl')}
        domain = config['domain']
        for evaluation in read(path/'evaluations.jsonl'):
            if evaluation['step'] not in checkpoints:
                continue
            candidates[domain].append({'source_run':path.name, 'step':evaluation['step'],
                'evaluation':evaluation, 'checkpoint':checkpoints[evaluation['step']]})
    selected = {}
    for domain, rows in candidates.items():
        if not rows:
            raise RuntimeError(f'No saved, evaluated teacher for {domain}')
        selected[domain] = choose(rows,domain,targets[domain])
    paths = {d:r['checkpoint']['sampler_path'] for d,r in selected.items()}
    assert len(set(paths.values()))==2
    report = {'rule':'First scheduled checkpoint reaching provisional goal; otherwise highest own-domain dev score, earliest on ties.',
        'original_teacher_quality_match_verified':False,
        'target_source':'Operational goals; original teacher scores unavailable. Not claimed as original teacher benchmarks.',
        'selected':selected}
    for filename, value in [('teacher-selection.json',report),('teachers.json',paths)]:
        with (root/filename).open('x') as f:
            json.dump(value,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
