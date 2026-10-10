"""verl entry point: `node.py` runs the controller, `node.py ROLE` runs one node.

verl runs on the hosts from the campaign's `source/.venv`. Every role joins one Ray cluster and verl
places its own resource pools on it: the trainer on one node, the policy engines and frozen teachers
on the generation node (b200x2) or nodes (h100x3, one teacher per node). The learner role runs the
driver (`train.py`). Each node's capture also scrapes the vLLM servers Ray starts there.
"""

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from shared import recipe
from shared.slurm import (CACHE, CACHE_ENV, PYDEPS, ROOT, Allocation, Supervisor, STOP_FILE, exit_on_signals, generation_hosts,
                          result_dir, site, write_json)

RAY_PORT = 6379


def generation_roles(s):
    """{role: (node, IP)} for the generation nodes: `generation`, then `generation_2` (h100x3)."""
    return {f'generation{"_" + str(i + 1) if i else ""}': host for i, host in enumerate(generation_hosts(s))}


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
           # Ray workers already use the venv's Python via `ray start`. Ray's `uv run` hook would require
           # pyproject.toml in the working directory and ship it to every worker.
           'RAY_ENABLE_UV_RUN_RUNTIME_ENV': '0'}

    def freeze_packages():
        with (result / 'installed-packages.txt').open('wb') as packages:
            subprocess.run([s['uv'], 'pip', 'freeze', '--python', str(ROOT / 'source/.venv/bin/python')],
                           stdout=packages, check=True)

    srun = ['srun', '--exclusive', '-N1', '-n1', f'--gres=gpu:{recipe.GPUS_PER_NODE}',
            f'--cpus-per-task={recipe.CPUS_PER_NODE}']
    nodes = {'learner': s['trainer_node'], **{role: node for role, (node, _) in generation_roles(s).items()}}
    steps = {role: [*srun, '-w', node, *uv_python(s), str(ROOT / 'node.py'), role] for role, node in nodes.items()}
    return allocation.run(steps, env, before=freeze_packages)


def wait_for_gpus(head, processes, timeout=300):
    """Start the driver only once every node's GPUs are in the Ray cluster."""
    import ray
    ray.init(address=f'{head}:{RAY_PORT}')
    deadline = time.monotonic() + timeout
    while int(ray.cluster_resources().get('GPU', 0)) != (1 + recipe.GENERATION_NODES) * recipe.GPUS_PER_NODE:
        processes.check()
        if time.monotonic() > deadline:
            raise TimeoutError(f'Ray has {ray.cluster_resources().get("GPU", 0)} GPUs')
        time.sleep(2)
    ray.shutdown()


def keep_trainer_log(result, role):
    """Copy the trainer actor's Ray worker log, which holds verl's per-step console metrics, into the result.

    Ray keeps worker logs on the node that ran them and does not always forward them to the driver.
    """
    try:
        for log in Path('/tmp/ray/session_latest/logs').glob('worker-*.out'):
            if b'training/global_step' in log.read_bytes():
                shutil.copyfile(log, result / f'{role}-trainer-worker.out')
    except OSError as error:
        print(f'could not copy the trainer log: {error}', file=sys.stderr)


def main(role):
    s, result = site(), result_dir()
    generation = generation_roles(s)
    ip = s['trainer_ip'] if role == 'learner' else generation[role][1]
    processes = Supervisor(result, prefix=f'{role}-')
    exit_on_signals()
    # Slurm's GPU plugin also sets the AMD variable, and verl workers refuse to start if both are set.
    os.environ.pop('ROCR_VISIBLE_DEVICES', None)
    ray = ['ray', 'start', f'--node-ip-address={ip}', f'--num-gpus={recipe.GPUS_PER_NODE}',
           f'--num-cpus={recipe.RAY_CPUS_PER_NODE}', '--disable-usage-stats', '--block']
    # `--vllm` scrapes the Prometheus endpoint of each vLLM server (policy or teacher) on this node.
    capture = [sys.executable, '-m', 'shared.capture', role, '--vllm']
    # Ray workers inherit this. Without it the trainer can run out of memory from fragmentation at
    # 32k, and verl already enables it for its vLLM servers.
    ray_env = {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', **CACHE_ENV}
    if recipe.H100:
        # On the CoreWeave NFS share, vLLM engines on different nodes compiling into one shared cache hit
        # "Stale file handle"; the failing TP rank's partner then waited forever (job 542). Keep the
        # compiled-kernel caches on each node's local disk instead.
        local = Path('/tmp/opd-runtime-cache') / ROOT.name
        ray_env.update({key: str(local / Path(value).relative_to(CACHE)) if Path(value) != CACHE else str(local)
                        for key, value in CACHE_ENV.items()})
    if recipe.TRAINER_OPTIMIZER_CPU_OFFLOAD:
        # Ray gives each worker OMP_NUM_THREADS=1 unless it is set, which runs the CPU optimizer step on one
        # core per rank (as in Miles run 532). Ray places verl's trainer pool on any node, so set it everywhere.
        ray_env['OMP_NUM_THREADS'] = str(recipe.CPUS_PER_NODE // recipe.TRAINER_GPUS)
    try:
        processes.launch(capture, 'capture')
        if role == 'learner':
            processes.launch([*ray, '--head'], 'ray', ray_env)
            write_json(result / 'head-started.json', {'ip': ip})
            for name in generation:
                processes.wait_for_file(f'{name}-ready.json')
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
        write_json(result / f'{role}-ready.json', {'ip': ip})
        while not (result / STOP_FILE).exists():
            processes.check()
            time.sleep(3)
        return 0
    finally:
        keep_trainer_log(result, role)
        processes.stop_all()


if __name__ == '__main__':
    sys.exit(control() if len(sys.argv) == 1 else main(sys.argv[1]))
