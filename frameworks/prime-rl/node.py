"""Prime-RL campaign entry point: `node.py` runs the controller, `node.py ROLE` one node.

Prime-RL runs from the campaign's `source/.venv` on the hosts, not in a
container. The generation node serves the frozen teachers, three policy
engines and their router; the trainer node runs the trainer, env servers and
orchestrator. `make_config.py` holds the recipe and ports.
"""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

from shared import recipe
from shared.slurm import CACHE_ENV, PYDEPS, ROOT, Allocation, Supervisor, exit_on_signals, result_dir, site
import make_config as config

FINISH_TIMEOUT = 900  # Allowed gap between the trainer and orchestrator finishing.


def uv_python(s):
    # --no-sync keeps a run from silently replacing the provisioned environment.
    return [s['uv'], 'run', '--project', str(ROOT / 'source'), '--no-sync', 'python']


def control():
    allocation = Allocation('prime-rl')
    s, result = allocation.site, allocation.result
    allocation.start(learner_ip=s['trainer_ip'], generation_ip=s['generation_ip'])
    env = {**os.environ, 'CAMPAIGN_RESULT': str(result), 'PRL_RUN_ID': result.name, 'WANDB_MODE': 'disabled',
           'PYTHONPATH': str(ROOT / 'environments/rg_tasks'), 'CAMPAIGN_PYDEPS': str(PYDEPS),
           'UV_CACHE_DIR': str(ROOT / 'uv-cache'),
           'UV_NO_SYNC': '1'}

    def resolve_configs():
        with (result / 'installed-packages.txt').open('wb') as packages:
            subprocess.run([s['uv'], 'pip', 'freeze', '--python', str(ROOT / 'source/.venv/bin/python')],
                           stdout=packages, check=True)
        with (result / 'make-config.out').open('wb') as log:
            subprocess.run([*uv_python(s), str(ROOT / 'make_config.py'), str(result)], env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)

    srun = ['srun', '--exclusive', '-N1', '-n1', f'--gres=gpu:{recipe.GPUS_PER_NODE}',
            f'--cpus-per-task={recipe.CPUS_PER_NODE}']
    steps = {role: [*srun, '-w', s[node], *uv_python(s), str(ROOT / 'node.py'), role]
             for role, node in (('generation', 'generation_node'), ('learner', 'trainer_node'))}
    return allocation.run(steps, env, before=resolve_configs)


class Node:
    def __init__(self, role):
        from prime_rl.utils.process import DEFAULT_COMMON_ENV_VARS
        self.role, self.site, self.result = role, site(), result_dir()
        paths = json.loads((self.result / 'resolved-paths.json').read_text())
        self.config, self.logs = Path(paths['config']), Path(paths['logs'])
        env = {**os.environ, **DEFAULT_COMMON_ENV_VARS, 'WANDB_MODE': 'disabled', 'NCCL_DEBUG': 'WARN',
               'PRL_ATTEMPT_CONFIG_DIR': str(self.config), 'PRL_ATTEMPT_LOG_DIR': str(self.logs), **CACHE_ENV}
        # Component logs share one directory across both nodes, so names are unique per node.
        self.processes = Supervisor(self.result, base_env=env)

    def launch(self, command, name, env=None):
        full = [self.site['uv'], 'run', '--project', ROOT / 'source', '--no-sync', *command]
        return self.processes.launch(full, name, env, log=self.logs / f'{name}.log')

    def inference(self, name, gpus):
        from prime_rl.utils.process import DEFAULT_INFERENCE_ENV_VARS
        # Separate config and log directories keep standalone inference launchers isolated.
        root = self.result / name
        self.launch(['inference', '@', self.config / f'{name}.json'], name,
                    {**DEFAULT_INFERENCE_ENV_VARS, 'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus)),
                     'PRL_ATTEMPT_CONFIG_DIR': str(root / 'configs'), 'PRL_ATTEMPT_LOG_DIR': str(root / 'logs')})

    def generation(self):
        for domain, gpu in recipe.TEACHER_GPU.items():
            self.inference(f'teacher-{domain}', [gpu])
        for index in range(len(config.POLICY_PORTS)):
            self.inference(f'policy-{index}', config.policy_gpus(index))
        # Upstream's router arguments (`start_router` in prime_rl/entrypoints/inference.py), over the three engines.
        self.launch(['vllm-router', '--policy', 'consistent_hash', '--host', '0.0.0.0', '--port', config.ROUTER_PORT,
                     '--worker-urls', *(f'http://127.0.0.1:{port}' for port in config.POLICY_PORTS),
                     '--intra-node-data-parallel-size', '1', '--request-id-headers', 'x-session-id',
                     '--worker-startup-timeout-secs', '4200', '--prometheus-port', config.ROUTER_METRICS_PORT],
                    'policy-router')
        while True:
            self.processes.check()
            time.sleep(5)

    def learner(self):
        from prime_rl.utils.process import DEFAULT_TRAINER_ENV_VARS
        trainer = self.launch(['torchrun', '--standalone', f'--nproc-per-node={recipe.TRAINER_GPUS}',
                               f'--log-dir={self.logs / "trainer-ranks"}', '--redirect=3', '--tee=3',
                               '-m', 'prime_rl.trainer.rl.train', '@', self.config / 'trainer.json'], 'trainer',
                              {**DEFAULT_TRAINER_ENV_VARS,
                               'CUDA_VISIBLE_DEVICES': ','.join(map(str, range(recipe.TRAINER_GPUS)))})
        for path in sorted((self.config / 'envs').glob('*/*.json')):
            self.launch(['env-server', '@', path], f'env-{path.parent.name}-{path.stem}')
        self.wait_for_servers()
        orchestrator = self.launch(['orchestrator', '@', self.config / 'orchestrator.json'], 'orchestrator')
        first_finished = None
        while True:
            self.processes.check(allowed=('trainer', 'orchestrator'))
            states = [trainer.poll(), orchestrator.poll()]
            if any(state not in (None, 0) for state in states):
                raise RuntimeError(f'Trainer/orchestrator exited unsuccessfully: {states}')
            if states == [0, 0]:
                return 0
            if 0 in states:
                first_finished = first_finished or time.monotonic()
                if time.monotonic() - first_finished > FINISH_TIMEOUT:
                    raise TimeoutError('The remaining training component did not finish in time')
            time.sleep(5)

    def wait_for_servers(self, timeout=1500):
        pending = {config.ROUTER_PORT, *config.INFERENCE_PORTS}
        deadline = time.monotonic() + timeout
        while pending:
            self.processes.check()
            for port in list(pending):
                try:
                    with urllib.request.urlopen(f'http://{self.site["generation_ip"]}:{port}/health', timeout=2):
                        pending.remove(port)
                except OSError:
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError(f'Servers did not become ready: {pending}')
            time.sleep(3)

    def main(self):
        exit_on_signals()
        # Fail before loading models if a campaign port is occupied.
        for port in config.GENERATION_NODE_PORTS if self.role == 'generation' else config.TRAINER_NODE_PORTS:
            with socket.socket() as probe:
                probe.bind(('0.0.0.0', port))
        targets = ([f'127.0.0.1:{p}' for p in (*config.INFERENCE_PORTS, config.ROUTER_METRICS_PORT)]
                   if self.role == 'generation' else [])
        self.launch(['python', '-m', 'shared.capture', self.role, *targets], f'{self.role}-capture',
                    {'PYTHONPATH': f'{ROOT}:{os.environ.get("PYTHONPATH", "")}'})
        try:
            return self.generation() if self.role == 'generation' else self.learner()
        finally:
            self.processes.stop_all(timeout=15)


if __name__ == '__main__':
    sys.exit(control() if len(sys.argv) == 1 else Node(sys.argv[1]).main())
