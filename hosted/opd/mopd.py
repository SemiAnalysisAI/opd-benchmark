"""Multi-teacher on-policy distillation (MOPD): one fresh Qwen3.6-35B-A3B student learns several domains
at once, each from its own frozen teacher.

Every update takes an equal share of `--prompts` from each domain; each sampled response is
scored by the teacher of its own domain, and the student takes one step on all of them
together (common.py describes the update). Thinking is on.

    python hosted/opd/mopd.py --teachers teachers.json --output /abs/mopd

teachers.json maps each domain to its teacher's sampler path, for example the one
hosted/tinker/launch.py --teachers sft writes. Its domains are the ones trained.
"""
import argparse
import json
from pathlib import Path

from common import Experiment, add_protocol_arguments, check_protocol_arguments, run
from shared import recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--teachers', type=Path, required=True, help='JSON: {domain: teacher sampler path}')
    add_protocol_arguments(parser)
    args = parser.parse_args()
    check_protocol_arguments(parser, args)
    teachers = json.loads(args.teachers.read_text())
    if len(teachers) < 2 or not set(teachers) <= set(recipe.DOMAINS):
        parser.error(f'--teachers needs two or more of {recipe.DOMAINS}; use opd.py for one')
    if args.prompts % len(teachers):
        parser.error('--prompts must split evenly over the domains')
    share = args.prompts // len(teachers)
    domains = [d for d in recipe.DOMAINS if d in teachers]  # A fixed order, whatever the file's.
    run(lambda: Experiment(args, 'mopd', {d: teachers[d] for d in domains}),
        lambda experiment, update: [p for d in domains for p in experiment.train_prompts(d, update, share)])


if __name__ == '__main__':
    main()
