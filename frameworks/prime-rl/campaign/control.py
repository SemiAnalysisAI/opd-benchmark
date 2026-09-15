"""Own the Slurm allocation and launch native Prime-RL components on two nodes."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
UV = '@UV@'


def stop(*_):
    raise SystemExit(143)


def main():
    job = os.environ['SLURM_JOB_ID']
    result = ROOT / 'results' / f'prime-opd-{job}'
    result.mkdir(parents=True, exist_ok=False)
    nodes = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']], text=True).split()
    assert nodes == ['@GENERATION_NODE@', '@TRAINER_NODE@'], nodes
    generation_ip, learner_ip = '@GENERATION_IP@', '@TRAINER_IP@'
    env = {**os.environ, 'PRL_RUN_ID': f'prime-opd-{job}', 'WANDB_MODE': 'disabled',
           'PYTHONPATH': str(ROOT / 'environments/mopd_puzzles'),
           'UV_CACHE_DIR': str(ROOT / 'uv-cache'), 'UV_NO_SYNC': '1'}
    (ROOT / 'active-run.json').write_text(json.dumps({'job_id': job, 'result': str(result)}))
    (result / 'start.json').write_text(json.dumps({'job_id': job, 'time': time.time(), 'nodes': nodes,
        'learner_ip': learner_ip, 'generation_ip': generation_ip,
        'physical_gpu_roles': {'@TRAINER_NODE@': {'trainer': list(range(8))},
                              '@GENERATION_NODE@': {'countdown_teacher': [0], 'graph_color_teacher': [1], 'rollout': list(range(2, 8))}}}))
    (result / 'slurm-start.txt').write_bytes(subprocess.check_output(['scontrol', 'show', 'job', job]))
    subprocess.run(['tar', '--exclude=.venv', '--exclude=.git', '--exclude=__pycache__', '-czf', str(result / 'launch-source.tar.gz'),
                    '-C', str(ROOT), 'source', 'data', 'environments', 'make_config.py', 'control.py', 'node.py',
                    'telemetry.py', 'preflight.py', 'teacher-provenance.json', 'source-provenance.json'], check=True)
    children = []
    code = 1
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        with (result / 'config-preflight.out').open('wb') as log:
            subprocess.run([UV, 'pip', 'freeze', '--python', str(ROOT / 'source/.venv/bin/python')],
                           stdout=(result / 'installed-packages.txt').open('wb'), check=True)
            subprocess.run([UV, 'run', '--project', str(ROOT / 'source'), '--no-sync', 'python', str(ROOT / 'make_config.py'),
                            str(result), learner_ip, generation_ip], env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
            subprocess.run([UV, 'run', '--project', str(ROOT / 'source'), '--no-sync', 'python', str(ROOT / 'preflight.py')],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        for role, node in [('generation', nodes[0]), ('learner', nodes[1])]:
            command = ['srun', '--exclusive', '-N1', '-n1', '--gres=gpu:8', '--cpus-per-task=80', '-w', node,
                       UV, 'run', '--project', str(ROOT / 'source'), '--no-sync', 'python', str(ROOT / 'node.py'), str(result), role]
            (result / f'{role}-srun.json').write_text(json.dumps(command))
            children.append(subprocess.Popen(command, env=env, stdout=(result / f'{role}-driver.out').open('wb'),
                                             stderr=subprocess.STDOUT, start_new_session=True))
        while True:
            if children[1].poll() is not None:
                code = children[1].returncode
                break
            if children[0].poll() is not None:
                raise RuntimeError(f'Generation exited early with code {children[0].returncode}')
            time.sleep(5)
    finally:
        (result / 'STOP').touch()
        for process in children:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        (result / 'end.json').write_text(json.dumps({'time': time.time(), 'exit_code': code}))
        (result / 'exit-code.txt').write_text(f'{code}\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
