"""NeMo-RL campaign entry point: `node.py` runs the controller inside the Slurm allocation.

NeMo-RL brings up its multi-node Ray cluster with upstream `ray.sub` (in the pinned source) and places
its own trainer, vLLM and teacher worker groups on it, so there are no per-role steps here. The
controller writes the config (`make_config.py`), starts the package's telemetry capture on every node,
runs `ray.sub` with the driver command, and records the outcome like every other framework.
"""
import json
import os
from pathlib import Path
import sys
import time

from shared import recipe
from shared.slurm import CACHE_ENV, PYDEPS, ROOT, STOP_FILE, Allocation, Supervisor, exit_on_signals, write_json
import make_config

# reasoning_gym is installed into NeMo Gym's server venv at runtime; pin the version every other
# framework's verifier and data use.
CONSTRAINTS = ROOT / 'nemo-gym-constraints.txt'


def driver_command(result):
    revision = json.loads((ROOT / 'package-provenance.json').read_text())['revision']
    return ' '.join([
        # The container's code must be the pinned revision.
        f'[ "$NEMO_RL_COMMIT" = {revision} ] && cd {make_config.UPSTREAM} &&', f'UV_CONSTRAINT={CONSTRAINTS}', 'HF_HUB_OFFLINE=1',
        # Unbuffered, so the driver's progress lines (including the weight syncs) reach the log as they happen.
        'PYTHONUNBUFFERED=1',
        # The image ships no NeMo Gym server venvs; build them once per campaign, not once per run.
        f'NEMO_GYM_VENV_DIR={ROOT / "runtime-cache" / "gym_venvs"}',
        'uv run examples/nemo_gym/run_grpo_nemo_gym.py', f'--config {result / "nemo-rl.yaml"}'])


def control():
    allocation = Allocation('nemo-rl')
    s, result = allocation.site, allocation.result
    allocation.start()
    make_config.main(result)
    CONSTRAINTS.write_text('reasoning-gym==0.1.25\n')
    nodes = [s['generation_node'], s['trainer_node'], s['teacher_node']]
    mounts = sorted({str(ROOT), s['assets'], str(Path(s['base_model']).parent)})
    env = {**os.environ, 'CAMPAIGN_RESULT': str(result), 'CAMPAIGN_PYDEPS': str(PYDEPS), 'PYTHONPATH': str(ROOT),
           # Slurm sets NVIDIA_VISIBLE_DEVICES=void; Pyxis needs these to expose the GPUs and driver libraries.
           'NVIDIA_VISIBLE_DEVICES': 'all', 'NVIDIA_DRIVER_CAPABILITIES': 'all',
           'CONTAINER': s['nemo_rl_container'], 'MOUNTS': ','.join(f'{m}:{m}' for m in mounts),
           'COMMAND': driver_command(result), 'BASE_LOG_DIR': str(result), 'GPUS_PER_NODE': str(recipe.GPUS_PER_NODE),
           # NeMo-RL writes into its image at runtime (uv syncs the project, vLLM files are patched in place), but
           # ray.sub's srun steps pass no --container-writable; this is Pyxis' environment form of that flag.
           # Not ray.sub's UV_CACHE_DIR_OVERRIDE: it mounts over /root/.cache/uv, where the image's venv files are
           # symlinked, and so empties the venv.
           'PYXIS_CONTAINER_WRITABLE': '1', 'HF_HOME': str(ROOT / 'runtime-cache' / 'hf'),
           **CACHE_ENV}
    exit_on_signals()
    processes = Supervisor(result)
    code = 1
    try:
        for node in nodes:  # Telemetry on each host; NeMo-RL decides which node runs what.
            processes.launch(['srun', '--overlap', '-N1', '-n1', '-w', node, sys.executable, '-m', 'shared.capture',
                              f'node-{node}'], f'capture-{node}', env=env)
        ray = processes.launch(['bash', ROOT / 'source' / 'ray.sub'], 'ray-sub', env=env)
        while ray.poll() is None:
            processes.check(allowed=('ray-sub',))
            time.sleep(10)
        code = ray.returncode
    finally:
        (result / STOP_FILE).touch()
        processes.stop_all(grace=30, timeout=20)
        (result / 'exit-code.txt').write_text(f'{code}\n')
        write_json(result / 'end.json', {'time': time.time(), 'exit_code': code})
    return code


if __name__ == '__main__':
    sys.exit(control())
