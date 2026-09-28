"""Print the salloc command for a prepared campaign; with --submit, check prerequisites and run it.

Copied into each campaign as `launch.py`.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path, help='prepared campaign directory')
    parser.add_argument('--partition', required=True, help='Slurm partition')
    # A 20-update run with 30k-token thinking takes about an hour after startup.
    parser.add_argument('--time', default='04:00:00', help='Slurm time limit (default: 04:00:00)')
    parser.add_argument('--submit', action='store_true', help='allocate GPUs instead of printing the command')
    args = parser.parse_args()
    root = args.campaign.resolve()
    site = json.loads((root/'site.json').read_text())
    manifest = json.loads((root/'package-provenance.json').read_text())
    name = 'opd-' + manifest['framework'] + '-' + hashlib.sha256(str(root).encode()).hexdigest()[:8]
    nodes = [site['generation_node'], site['trainer_node'], *([site['teacher_node']] if 'teacher_node' in site else [])]
    command = ['salloc', f'--job-name={name}', f'--partition={args.partition}', f'--nodes={len(nodes)}',
               f'--nodelist={",".join(nodes)}', '--exclusive',
               '--gres=gpu:8', f'--time={args.time}', 'python3', str(root/'node.py')]
    print(shlex.join(command), flush=True)
    if not args.submit:
        print('Dry run only. Add --submit to allocate GPUs.')
        return
    import fcntl
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'active-run.json').exists():
            previous = json.loads((root/'active-run.json').read_text())
            if not (Path(previous['result'])/'end.json').is_file():
                raise RuntimeError('The previous run has no terminal record. Inspect it before submitting again.')
        sys.path.insert(0, str(root))  # The campaign's own copy of the recipe.
        from shared.recipe import DOMAINS
        required = [Path(site['base_model'])/'config.json',
                    *(Path(site['teachers'])/domain/'config.json' for domain in DOMAINS)]
        if manifest['framework'] == 'nemo-rl':  # Own container; scores through NeMo Gym, so no pydeps.
            required += [Path(site['nemo_rl_container'])]
        elif manifest['framework'] in ('prime-rl', 'verl'):  # Host virtual environments.
            required += [root/'pydeps/reasoning_gym', root/'source/.venv/bin/python', Path(site['uv'])]
        else:
            required += [root/'pydeps/reasoning_gym', Path(site['container_image']), Path(site['megatron_model'])]
        if manifest['framework'] == 'slime':
            required += [root/line.split()[1] for line in (root/'wheels.sha256').read_text().splitlines()]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise RuntimeError(f'Missing prerequisites: {missing}')
        with (root/'allocation.out').open('ab', buffering=0) as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
