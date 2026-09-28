"""Two-node Pyxis container campaigns for Miles and Slime.

`node.py` with no argument is the controller: it holds the Slurm allocation
and starts one container step per node, `node.py learner` on the trainer node
and `node.py generation` on the generation node. The roles coordinate through
ready files in the result directory:

    learner (trainer node)            generation (generation node)
    ray head, head-started.json  -->  ray start --address=head
    (waits)                      <--  generation-ready.json
    placement-approved.json      -->  one frozen SGLang teacher per GPU
    (waits)                      <--  teachers-ready.json
    framework training driver
"""
import importlib
import ipaddress
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

from shared import recipe
from shared.slurm import (CACHE_ENV, PYDEPS, ROOT, STOP_FILE, Allocation, Supervisor, exit_on_signals, result_dir,
                          site, write_json)

TEACHER_PORTS = {domain: 30000 + i for i, domain in enumerate(recipe.DOMAINS)}
RAY_PORT = 6379
BASE_MODEL = ROOT / 'models' / recipe.BASE_MODEL_DIR


def entry(framework, node_class):
    """Run the controller for `node.py`, or the role for `node.py ROLE`."""
    return control(framework) if len(sys.argv) == 1 else node_class(sys.argv[1]).main()


def control(framework):
    allocation = Allocation(framework)
    allocation.start()
    s = allocation.site
    read_only = [s['assets'], str(Path(s['megatron_model']).parent), str(Path(s['base_model']).parent)]
    mounts = ','.join([f'{ROOT}:{ROOT}', *(f'{path}:{path}:ro' for path in read_only)])
    srun = ['srun', '--exclusive', '-N1', '-n1', f'--gres=gpu:{recipe.GPUS_PER_NODE}',
            f'--cpus-per-task={recipe.CPUS_PER_NODE}', f'--container-image={s["container_image"]}',
            '--container-writable', '--container-remap-root', '--no-container-mount-home',
            f'--container-mounts={mounts}']
    steps = {role: srun + ['-w', s[node], 'python3', str(ROOT / 'node.py'), role]
             for role, node in (('learner', 'trainer_node'), ('generation', 'generation_node'))}
    env = {**os.environ, 'CAMPAIGN_RESULT': str(allocation.result), 'CAMPAIGN_ROOT': str(ROOT),
           'NVIDIA_VISIBLE_DEVICES': 'all', 'NVIDIA_DRIVER_CAPABILITIES': 'all'}
    return allocation.run(steps, env)


class ContainerNode:
    """One role inside its container. Frameworks override the three hooks below."""

    placement_module = None  # Module that defines the framework's `_create_placement_group`.
    env = {}  # Extra environment for every process in the container.

    def __init__(self, role):
        assert role in ('learner', 'generation'), role
        self.role, self.site, self.result = role, site(), result_dir()
        self.job = os.environ['SLURM_JOB_ID']
        self.ip = self.site['trainer_ip' if role == 'learner' else 'generation_ip']
        self.processes = Supervisor(self.result, prefix=f'{role}-')

    def setup(self):
        """Adjust the container before anything starts."""

    def placement(self, group):
        """Return (sorted physical GPU ids, Ray placement group) for the framework's placement result."""
        raise NotImplementedError

    def train_command(self, teacher_urls):
        """Return (command, extra environment) for the framework's training driver."""
        raise NotImplementedError

    def main(self):
        os.chdir(ROOT / 'source')
        # The placement check imports the framework in this process; use the pinned source, not the image's copy.
        sys.path.insert(0, str(ROOT / 'source'))
        self.setup()
        os.environ.update({
            'PYTHONPATH': f'{ROOT}:{ROOT}/source:{self.site["megatron_source"]}', 'PYTHONNOUSERSITE': '1',
            'PYTHONUNBUFFERED': '1', 'WANDB_MODE': 'disabled', 'NCCL_DEBUG': 'WARN', 'MASTER_ADDR': self.ip,
            'RAY_TMPDIR': f'/tmp/campaign-ray-{self.job}-{self.role}', 'RAY_memory_monitor_refresh_ms': '0',
            'RAY_DEDUP_LOGS': '0', 'TENSORBOARD_DIR': str(self.result / 'tensorboard'), 'CAMPAIGN_PYDEPS': str(PYDEPS),
            **CACHE_ENV, **self.env})
        if self.role == 'learner':
            # Without this the Megatron trainer runs out of memory at 32k from fragmentation.
            # Trainer node only; the SGLang engines on the generation node keep the default allocator.
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ.pop('NETRC', None)
        exit_on_signals()
        (self.result / f'{self.role}-gpu-before.txt').write_bytes(subprocess.check_output(['nvidia-smi', '-q']))
        if self.role == 'learner':
            capture = ['--engine-log', self.result / 'learner-train.out', '--engine-host', self.site['generation_ip']]
        else:
            capture = [f'127.0.0.1:{port}' for port in TEACHER_PORTS.values()]
        self.processes.launch([sys.executable, '-m', 'shared.capture', self.role, *capture], 'capture')
        try:
            return self.learner() if self.role == 'learner' else self.generation()
        finally:
            self.processes.stop_all()
            for logs in Path('/tmp').glob(f'campaign-ray-{self.job}-*/ray/session_*/logs'):
                if not logs.parent.is_symlink():
                    shutil.copytree(logs, self.result / 'ray-logs' / self.role, dirs_exist_ok=True)

    def ray_start(self, *arguments, gpus, env=None):
        self.processes.launch(['ray', 'start', f'--node-ip-address={self.ip}', f'--num-gpus={gpus}',
                               f'--num-cpus={recipe.RAY_CPUS_PER_NODE}', '--disable-usage-stats', '--block',
                               *arguments], 'ray', env)

    def learner(self):
        self.ray_start('--head', gpus=recipe.TRAINER_GPUS)
        write_json(self.result / 'head-started.json', {'ip': self.ip})
        generation_ip = self.processes.wait_for_file('generation-ready.json')['ip']
        self.verify_placement(generation_ip)
        write_json(self.result / 'placement-approved.json', {'time': time.time()})
        self.processes.wait_for_file('teachers-ready.json')
        urls = {domain: f'http://{generation_ip}:{port}/generate' for domain, port in TEACHER_PORTS.items()}
        command, env = self.train_command(urls)
        self.processes.launch(command, 'train', env)
        return self.processes.run_until_exit('train')

    def generation(self):
        head = self.processes.wait_for_file('head-started.json')['ip']
        self.wait_until(lambda: socket.create_connection((head, RAY_PORT), timeout=2).close(), 'the Ray head')
        self.ray_start(f'--address={head}:{RAY_PORT}', gpus=len(recipe.POLICY_GPUS),
                       env={'CUDA_VISIBLE_DEVICES': ','.join(map(str, recipe.POLICY_GPUS))})
        write_json(self.result / 'generation-ready.json', {'ip': self.ip})
        self.processes.wait_for_file('placement-approved.json')
        for domain, port in TEACHER_PORTS.items():
            self.serve_teacher(domain, port)
        for port in TEACHER_PORTS.values():
            self.wait_until(lambda: urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5),
                            f'the teacher on port {port}', timeout=600)
        write_json(self.result / 'teachers-ready.json', {'ip': self.ip, 'ports': list(TEACHER_PORTS.values())})
        while not (self.result / STOP_FILE).exists():
            self.processes.check()
            time.sleep(3)
        return 0

    def serve_teacher(self, domain, port):
        """Start one frozen teacher. It only prefills student responses to score them; it never generates."""
        self.processes.launch([
            sys.executable, '-m', 'sglang.launch_server', '--model-path', ROOT / 'teachers' / domain,
            '--tokenizer-path', BASE_MODEL, '--host', '0.0.0.0', '--port', port, '--tp-size', '1',
            '--context-length', recipe.CONTEXT_LENGTH, '--mem-fraction-static', '0.8',
            '--max-running-requests', '128', '--disable-cuda-graph', '--chunked-prefill-size', '-1',
            '--disable-radix-cache', '--prefill-max-requests', '1', '--moe-runner-backend', 'triton',
            '--enable-metrics'], f'teacher-{domain}', {'CUDA_VISIBLE_DEVICES': str(recipe.TEACHER_GPU[domain])})

    def wait_until(self, attempt, what, timeout=180):
        """Retry `attempt` until it stops raising OSError, failing fast if a child exits."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                return attempt()
            except OSError:
                self.processes.check()
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Timed out waiting for {what}')
                time.sleep(2)

    def verify_placement(self, generation_ip):
        """Check that the framework places the trainer and policy on the intended physical GPUs."""
        import ray
        from ray.util import remove_placement_group

        total = recipe.TRAINER_GPUS + len(recipe.POLICY_GPUS)
        ray.init(address=f'{self.ip}:{RAY_PORT}')
        deadline = time.monotonic() + 180
        while int(ray.cluster_resources().get('GPU', 0)) != total:
            self.processes.check()
            assert time.monotonic() < deadline, ray.cluster_resources()
            time.sleep(2)
        nodes = {n['NodeManagerAddress']: n['Resources'].get('GPU') for n in ray.nodes() if n['Alive']}
        assert nodes == {self.ip: float(recipe.TRAINER_GPUS), generation_ip: float(len(recipe.POLICY_GPUS))}, nodes
        # The frameworks sort bundles by node IP, so the trainer must have the lower address.
        assert ipaddress.ip_address(self.ip) < ipaddress.ip_address(generation_ip), nodes
        gpu_ids, group = self.placement(importlib.import_module(self.placement_module)._create_placement_group(total))
        try:
            assert gpu_ids == list(range(recipe.TRAINER_GPUS)) + list(recipe.POLICY_GPUS), gpu_ids
            write_json(self.result / 'placement-verified.json', {'nodes': nodes, 'sorted_physical_gpu_ids': gpu_ids})
        finally:
            remove_placement_group(group)
            ray.shutdown()
