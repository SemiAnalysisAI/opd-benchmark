"""On-policy distillation (OPD): a fresh Qwen3.6-35B-A3B student learns one domain from one frozen teacher.

Every update the student answers `--prompts` training problems of the domain, the teacher
scores each sampled token, and the student takes one step towards the teacher on those
tokens (common.py describes the update). Thinking is on.

    python hosted/opd/opd.py --domain caesar_cipher --output /abs/opd-caesar \\
        --teacher tinker://.../sampler_weights/final

Evaluation covers every recipe domain, so it also shows what happens to the domain not trained on.
"""
import argparse

from common import Experiment, add_protocol_arguments, check_protocol_arguments, run
from shared import recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--domain', choices=recipe.DOMAINS, required=True)
    parser.add_argument('--teacher', required=True, help="The teacher's sampler path")
    add_protocol_arguments(parser)
    args = parser.parse_args()
    check_protocol_arguments(parser, args)
    run(lambda: Experiment(args, 'opd', {args.domain: args.teacher}),
        lambda experiment, update: experiment.train_prompts(args.domain, update, args.prompts))


if __name__ == '__main__':
    main()
