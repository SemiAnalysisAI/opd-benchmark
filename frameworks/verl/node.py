"""verl campaign entry point: `node.py` runs the controller, `node.py ROLE` one node.

verl runs from the campaign's `source/.venv` on the hosts. Both roles join one Ray
cluster; verl then places its own resource pools on it: the trainer (8 GPUs) on one
node, and the policy engines (6 GPUs) and the two frozen teachers (1 GPU each) on the
other. The learner role runs the driver (`train.py`).
"""

import os
import socket
import subprocess
import sys
import time

from shared import recipe
from shared.slurm import CACHE_ENV, PYDEPS, ROOT, Allocation, Supervisor, STOP_FILE, exit_on_signals, result_dir, site, write_json

RAY_PORT = 6379


def uv_python(s):
    # --no-sync keeps a run from silently replacing the provisioned environment.
    return [s['uv'], 'run', '--project', str(ROOT / 'source'), '--no-sync', 'python']


def control():
    allocation = Allocation('verl')
    s, result = allocation.site, allocation.result
    allocation.start()
    env = {**os.environ, 'CAMPAIGN_RESULT': str(result), 'CAMPAIGN_PYDEPS': str(PYDEPS),
           'PYTHONPATH': str(ROOT), 'UV_CACHE_DIR': str(ROOT / 'uv-cache'), 'UV_NO_SYNC': '1',
           'WANDB_MODE': 'disabled', 'CUDA_DEVICE_MAX_CONNECTIONS': '1',
           # Ray workers already run the venv's Python from `ray start`; Ray's `uv run` hook would instead
           # require the working directory to hold pyproject.toml and ship it to every worker.
           'RAY_ENABLE_UV_RUN_RUNTIME_ENV': '0'}

    def freeze_packages():
        with (result / 'installed-packages.txt').open('wb') as packages:
            subprocess.run([s['uv'], 'pip', 'freeze', '--python', str(ROOT / 'source/.venv/bin/python')],
                           stdout=packages, check=True)

    srun = ['srun', '--exclusive', '-N1', '-n1', f'--gres=gpu:{recipe.GPUS_PER_NODE}',
            f'--cpus-per-task={recipe.CPUS_PER_NODE}']
    steps = {role: [*srun, '-w', s[node], *uv_python(s), str(ROOT / 'node.py'), role]
             for role, node in (('learner', 'trainer_node'), ('generation', 'generation_node'))}
    return allocation.run(steps, env, before=freeze_packages)


def wait_for_gpus(head, processes, timeout=300):
    """Start the driver only once both nodes' GPUs are in the Ray cluster."""
    import ray
    ray.init(address=f'{head}:{RAY_PORT}')
    deadline = time.monotonic() + timeout
    while int(ray.cluster_resources().get('GPU', 0)) != 2 * recipe.GPUS_PER_NODE:
        processes.check()
        if time.monotonic() > deadline:
            raise TimeoutError(f'Ray has {ray.cluster_resources().get("GPU", 0)} GPUs')
        time.sleep(2)
    ray.shutdown()


def main(role):
    s, result = site(), result_dir()
    ip = s['trainer_ip' if role == 'learner' else 'generation_ip']
    processes = Supervisor(result, prefix=f'{role}-')
    exit_on_signals()
    # Slurm's GPU plugin also sets the AMD variable; verl workers refuse to start when it is set with CUDA's.
    os.environ.pop('ROCR_VISIBLE_DEVICES', None)
    ray = ['ray', 'start', f'--node-ip-address={ip}', f'--num-gpus={recipe.GPUS_PER_NODE}',
           f'--num-cpus={recipe.RAY_CPUS_PER_NODE}', '--disable-usage-stats', '--block']
    capture = [sys.executable, '-m', 'shared.capture', role]
    # Ray workers inherit this from their node. At 32k the trainer can run out of memory from
    # fragmentation without it, and verl enables it for its vLLM servers anyway.
    ray_env = {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', **CACHE_ENV}
    try:
        processes.launch(capture, 'capture')
        if role == 'learner':
            processes.launch([*ray, '--head'], 'ray', ray_env)
            write_json(result / 'head-started.json', {'ip': ip})
            processes.wait_for_file('generation-ready.json')
            wait_for_gpus(ip, processes)
            processes.launch([sys.executable, str(ROOT / 'train.py')], 'train', {'RAY_ADDRESS': f'{ip}:{RAY_PORT}'})
            return processes.run_until_exit('train')
        head = processes.wait_for_file('head-started.json')['ip']
        deadline = time.monotonic() + 180
        while True:  # The head writes its address before its GCS socket accepts connections.
            try:
                socket.create_connection((head, RAY_PORT), timeout=2).close()
                break
            except OSError:
                processes.check()
                if time.monotonic() > deadline:
                    raise TimeoutError('The Ray head did not accept connections')
                time.sleep(2)
        processes.launch([*ray, f'--address={head}:{RAY_PORT}'], 'ray', ray_env)
        write_json(result / 'generation-ready.json', {'ip': ip})
        while not (result / STOP_FILE).exists():
            processes.check()
            time.sleep(3)
        return 0
    finally:
        processes.stop_all()


if __name__ == '__main__':
    sys.exit(control() if len(sys.argv) == 1 else main(sys.argv[1]))
