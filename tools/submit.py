"""Print a Slurm command by default; submit only when --submit is explicit."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--partition', required=True)
    parser.add_argument('--time', default='01:30:00')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    root = args.campaign.resolve()
    site = json.loads((root/'site.json').read_text())
    manifest = json.loads((root/'package-provenance.json').read_text())
    name = 'opd-' + manifest['framework'] + '-' + hashlib.sha256(str(root).encode()).hexdigest()[:8]
    command = ['salloc', f'--job-name={name}', f'--partition={args.partition}', '--nodes=2',
               f'--nodelist={site["generation_node"]},{site["trainer_node"]}', '--exclusive',
               '--gres=gpu:8', f'--time={args.time}', 'python3', str(root/'control.py')]
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
        required = [Path(site['base_model'])/'config.json', Path(site['teachers'])/'countdown/config.json',
                    Path(site['teachers'])/'graph_color/config.json']
        if manifest['framework'] == 'prime-rl':
            required += [root/'source/.venv/bin/python', Path(site['base_model'])/'prime/.prime-v1', Path(site['uv'])]
        else:
            required += [Path(site['container_image']), Path(site['megatron_model'])]
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
