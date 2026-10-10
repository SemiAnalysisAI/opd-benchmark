"""Check that a backend's teachers match the Tinker teachers on recorded rollouts, before paying for a run.

Each reference rollout holds its prompt and response tokens, the Tinker teacher's logprobs, and the
near-base student's logprobs on the same tokens. A teacher passes when its logprobs are much closer
to the Tinker teacher's than the student's are (the ratio is at most --max-ratio).

    FIREWORKS_LEASE=/abs/lease python hosted/opd/check_teachers.py --backend fireworks \\
        --teachers /abs/lease/teachers.json --rollouts reference-rollouts.json --output /abs/lease/teacher-check.json
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import tinker

from backend import BACKENDS


def mean_abs(a, b):
    assert len(a) == len(b)
    return statistics.mean(abs(x - y) for x, y in zip(a, b))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backend', choices=sorted(BACKENDS), required=True)
    p.add_argument('--teachers', type=Path, required=True, help='JSON: {domain: teacher path}')
    p.add_argument('--rollouts', type=Path, required=True, help='JSON: {domain: [recorded rollouts]}')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-ratio', type=float, default=0.6)
    args = p.parse_args()
    teachers = json.loads(args.teachers.read_text())
    rollouts = json.loads(args.rollouts.read_text())
    backend = BACKENDS[args.backend](experiment='check-teachers')
    results, passed = [], True
    try:
        for domain, path in teachers.items():
            sampler = backend.sampler(path=path)
            for r in rollouts[domain]:
                start = time.monotonic()
                full = sampler.compute_logprobs(tinker.ModelInput.from_ints(r['prompt_tokens'] + r['tokens'])).result()
                teacher = full[len(r['prompt_tokens']):]
                gap, student_gap = mean_abs(teacher, r['teacher_logprobs']), mean_abs(r['student_logprobs'], r['teacher_logprobs'])
                ok = gap <= args.max_ratio * student_gap
                passed &= ok
                results.append({'domain': domain, 'index': r['index'], 'tokens': len(r['tokens']), 'passed': ok,
                                'mean_abs_teacher_minus_tinker_teacher': round(gap, 4),
                                'mean_abs_student_minus_tinker_teacher': round(student_gap, 4),
                                'seconds': round(time.monotonic() - start, 1)})
                print(json.dumps(results[-1]), flush=True)
    finally:
        backend.close()
    args.output.write_text(json.dumps({'passed': passed, 'max_ratio': args.max_ratio, 'results': results}, indent=2))
    if not passed:
        raise SystemExit('A teacher does not match its Tinker reference')


if __name__ == '__main__':
    main()
