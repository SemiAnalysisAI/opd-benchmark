"""Supervise one node's owned processes and collect lightweight measurements."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from prime_rl.utils.process import DEFAULT_COMMON_ENV_VARS, DEFAULT_INFERENCE_ENV_VARS, DEFAULT_TRAINER_ENV_VARS

ROOT = Path(__file__).resolve().parent
result, role = Path(sys.argv[1]), sys.argv[2]
paths = json.loads((result / 'resolved-paths.json').read_text())
config, logs = Path(paths['config']), Path(paths['logs'])
UV = '@UV@'
children = []
env = {**os.environ, **DEFAULT_COMMON_ENV_VARS, 'WANDB_MODE': 'disabled',
       'PRL_ATTEMPT_CONFIG_DIR': str(config), 'PRL_ATTEMPT_LOG_DIR': str(logs),
       'NCCL_DEBUG': 'WARN', 'TORCHINDUCTOR_CACHE_DIR': str(ROOT / 'runtime-cache' / role / 'inductor'),
       'TRITON_CACHE_DIR': str(ROOT / 'runtime-cache' / role / 'triton')}


def launch(command, name, extra=None):
    full = [UV, 'run', '--project', str(ROOT / 'source'), '--no-sync', *map(str, command)]
    (result / f'{role}-{name}-command.json').write_text(json.dumps(full))
    process = subprocess.Popen(full, env={**env, **(extra or {})}, stdout=(logs / f'{name}.log').open('wb'),
                               stderr=subprocess.STDOUT, start_new_session=True)
    children.append((name, process))
    return process


def check(allowed=()):
    if (result / 'STOP').exists():
        raise SystemExit(143)
    for name, process in children:
        if name not in allowed and process.poll() is not None:
            raise RuntimeError(f'{name} exited with {process.returncode}')


def stop(*_):
    raise SystemExit(143)


def wait_ready():
    generation_ip = json.loads((result / 'start.json').read_text())['generation_ip']
    deadline = time.monotonic() + 1500
    pending = {8000, 8100, 8110, 8120, 8200, 8300}
    while pending:
        check()
        for port in list(pending):
            try:
                with urllib.request.urlopen(f'http://{generation_ip}:{port}/health', timeout=2) as response:
                    if response.status == 200:
                        pending.remove(port)
            except OSError:
                pass
        if time.monotonic() > deadline:
            raise TimeoutError(f'Servers did not become ready: {pending}')
        time.sleep(3)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
code = 1
try:
    # Fail before loading models if a selected campaign port is occupied.
    ports = ((8000, 8100, 8110, 8120, 8200, 8300, 9100, 9110, 9120, 9200, 9300, 29000)
             if role == 'generation' else (5000, 5001, 5002, 5003, 15555, 15556, 29591))
    for port in ports:
        with socket.socket() as probe:
            probe.bind(('0.0.0.0', port))
    (result / f'{role}-ports-preflight.json').write_text(json.dumps({'time': time.time(), 'free_ports': ports}))
    launch(['python', ROOT / 'telemetry.py', result, role], f'{role}-telemetry')
    (result / f'{role}-gpu-before.txt').write_bytes(subprocess.check_output(['nvidia-smi', '-q']))
    if role == 'generation':
        for domain, gpu in [('countdown', '0'), ('graph_color', '1')]:
            # Separate logs and config directories keep standalone teacher launchers isolated.
            teacher_root = result / f'teacher-{domain}'
            launch(['inference', '@', config / f'teacher-{domain}.json'], f'teacher-{domain}',
                   {**DEFAULT_INFERENCE_ENV_VARS, 'CUDA_VISIBLE_DEVICES': gpu,
                    'PRL_ATTEMPT_CONFIG_DIR': str(teacher_root / 'configs'),
                    'PRL_ATTEMPT_LOG_DIR': str(teacher_root / 'logs')})
        for index in range(3):
            policy_root = result / f'policy-{index}'
            launch(['inference', '@', config / f'policy-{index}.json'], f'policy-{index}',
                   {**DEFAULT_INFERENCE_ENV_VARS,
                    'CUDA_VISIBLE_DEVICES': f'{2 + index * 2},{3 + index * 2}',
                    'PRL_ATTEMPT_CONFIG_DIR': str(policy_root / 'configs'),
                    'PRL_ATTEMPT_LOG_DIR': str(policy_root / 'logs')})
        # This is the native router command with three independent engine URLs.
        launch(['vllm-router', '--policy', 'power_of_two', '--host', '0.0.0.0',
                '--port', '8000', '--worker-urls',
                'http://127.0.0.1:8100', 'http://127.0.0.1:8110', 'http://127.0.0.1:8120',
                '--intra-node-data-parallel-size', '1', '--request-id-headers', 'x-session-id',
                '--worker-startup-timeout-secs', '4200', '--prometheus-port', '29000'], 'policy-router')
        while True:
            check()
            time.sleep(5)
    else:
        trainer = launch(['torchrun', '--standalone', '--nproc-per-node=8',
                          f'--log-dir={logs / "trainer-ranks"}', '--redirect=3', '--tee=3',
                          '-m', 'prime_rl.trainer.rl.train', '@', config / 'trainer.json'],
                         'trainer', {**DEFAULT_TRAINER_ENV_VARS, 'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'})
        for path in sorted((config / 'envs').glob('*/*.json')):
            launch(['env-server', '@', path], f'env-{path.parent.name}-{path.stem}')
        wait_ready()
        (result / 'servers-ready.json').write_text(json.dumps({'time': time.time()}))
        orchestrator = launch(['orchestrator', '@', config / 'orchestrator.json'], 'orchestrator')
        completed_since = None
        while True:
            check(allowed=('trainer', 'orchestrator'))
            states = [trainer.poll(), orchestrator.poll()]
            if any(state not in (None, 0) for state in states):
                raise RuntimeError(f'Trainer/orchestrator exited unsuccessfully: {states}')
            if states == [0, 0]:
                code = 0
                break
            if any(state == 0 for state in states):
                completed_since = completed_since or time.monotonic()
                if time.monotonic() - completed_since > 900:
                    raise TimeoutError('The remaining training component did not finish within 15 minutes')
            time.sleep(5)
finally:
    for name, process in reversed(children):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for name, process in children:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    (result / f'{role}-component-exits.json').write_text(json.dumps({name: p.returncode for name, p in children}))
    (result / f'{role}-gpu-after.txt').write_bytes(subprocess.check_output(['nvidia-smi', '-q']))
sys.exit(code)
