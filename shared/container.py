"""Pyxis container campaigns for Miles and Slime: one trainer node and one or two generation nodes.

`node.py` with no argument is the controller: it holds the Slurm allocation
and starts one container step per node, `node.py learner` on the trainer node
and `node.py generation` on the generation node (plus `node.py generation-2` on
the second generation node in the h100x3 layout). The roles coordinate through
ready files in the result directory:

    learner (trainer node)            generation (each generation node)
    ray head, head-started.json  -->  ray start --address=head
    (waits)                      <--  generation-ready.json, generation-2-ready.json
    placement-approved.json      -->  this node's frozen SGLang teachers
    (waits)                      <--  teachers-ready.json, teachers-ready-2.json
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
from shared.slurm import (CACHE_ENV, PYDEPS, ROOT, STOP_FILE, Allocation, Supervisor, exit_on_signals,
                          generation_hosts, result_dir, site, write_json)

TEACHER_PORTS = {domain: 30000 + i for i, domain in enumerate(recipe.DOMAINS)}
# Generation roles in node order; role i runs on generation_hosts(site)[i].
GENERATION_ROLES = ('generation', 'generation-2')[:recipe.GENERATION_NODES]
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
    # --mem=0: the whole node's memory. Slurm's per-CPU default can be too small for CPU-offloaded Adam.
    srun = ['srun', '--exclusive', '-N1', '-n1', f'--gres=gpu:{recipe.GPUS_PER_NODE}', '--mem=0',
            f'--cpus-per-task={recipe.CPUS_PER_NODE}', f'--container-image={s["container_image"]}',
            '--container-writable', '--container-remap-root', '--no-container-mount-home',
            f'--container-mounts={mounts}']
    nodes = {'learner': s['trainer_node'],
             **{role: node for role, (node, _) in zip(GENERATION_ROLES, generation_hosts(s))}}
    steps = {role: srun + ['-w', node, 'python3', str(ROOT / 'node.py'), role] for role, node in nodes.items()}
    env = {**os.environ, 'CAMPAIGN_RESULT': str(allocation.result), 'CAMPAIGN_ROOT': str(ROOT),
           'NVIDIA_VISIBLE_DEVICES': 'all', 'NVIDIA_DRIVER_CAPABILITIES': 'all'}
    return allocation.run(steps, env)


class ContainerNode:
    """One role inside its container. Frameworks override the three hooks below."""

    placement_module = None  # Module that defines the framework's `_create_placement_group`.
    env = {}  # Extra environment for every process in the container.

    def __init__(self, role):
        assert role == 'learner' or role in GENERATION_ROLES, role
        self.role, self.site, self.result = role, site(), result_dir()
        self.job = os.environ['SLURM_JOB_ID']
        self.generation_ips = [ip for _, ip in generation_hosts(self.site)]
        # This generation node's index, which selects its teachers in recipe.TEACHER_PLACEMENT.
        self.index = GENERATION_ROLES.index(role) if role != 'learner' else None
        self.ip = self.site['trainer_ip'] if role == 'learner' else self.generation_ips[self.index]
        self.suffix = '' if not self.index else f'-{self.index + 1}'  # Ready-file suffix: '' or '-2'.
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
            # One cache per role: on NFS, a kernel one node rewrites becomes a stale file handle on another.
            **{key: f'{path}/{self.role}' for key, path in CACHE_ENV.items()}, **self.env})
        if self.role == 'learner':
            # Without this the Megatron trainer runs out of memory at 32k from fragmentation.
            # Trainer node only; the SGLang engines on the generation node keep the default allocator.
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ.pop('NETRC', None)
        exit_on_signals()
        (self.result / f'{self.role}-gpu-before.txt').write_bytes(subprocess.check_output(['nvidia-smi', '-q']))
        if self.role == 'learner':
            capture = ['--engine-log', self.result / 'learner-train.out', '--engine-host', *self.generation_ips]
        else:
            capture = [f'127.0.0.1:{TEACHER_PORTS[d]}' for d in self.teachers()]
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

    def teachers(self):
        """Domains whose teacher this generation node serves."""
        return [d for d, (node, _) in recipe.TEACHER_PLACEMENT.items() if node == self.index]

    def learner(self):
        self.ray_start('--head', gpus=recipe.TRAINER_GPUS)
        write_json(self.result / 'head-started.json', {'ip': self.ip})
        ips = [self.processes.wait_for_file(f'{role}-ready.json')['ip'] for role in GENERATION_ROLES]
        self.verify_placement(ips)
        write_json(self.result / 'placement-approved.json', {'time': time.time()})
        for i in range(len(GENERATION_ROLES)):
            self.processes.wait_for_file(f'teachers-ready{"" if not i else f"-{i + 1}"}.json')
        urls = {domain: f'http://{ips[node]}:{TEACHER_PORTS[domain]}/generate'
                for domain, (node, _) in recipe.TEACHER_PLACEMENT.items()}
        command, env = self.train_command(urls)
        self.processes.launch(command, 'train', env)
        return self.processes.run_until_exit('train')

    def generation(self):
        head = self.processes.wait_for_file('head-started.json')['ip']
        self.wait_until(lambda: socket.create_connection((head, RAY_PORT), timeout=2).close(), 'the Ray head')
        self.ray_start(f'--address={head}:{RAY_PORT}', gpus=len(recipe.POLICY_GPUS),
                       env={'CUDA_VISIBLE_DEVICES': ','.join(map(str, recipe.POLICY_GPUS))})
        write_json(self.result / f'{self.role}-ready.json', {'ip': self.ip})
        self.processes.wait_for_file('placement-approved.json')
        ports = [TEACHER_PORTS[domain] for domain in self.teachers()]
        for domain in self.teachers():
            self.serve_teacher(domain, TEACHER_PORTS[domain])
        for port in ports:
            self.wait_until(lambda: urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5),
                            f'the teacher on port {port}', timeout=900)
        write_json(self.result / f'teachers-ready{self.suffix}.json', {'ip': self.ip, 'ports': ports})
        while not (self.result / STOP_FILE).exists():
            self.processes.check()
            time.sleep(3)
        return 0

    def serve_teacher(self, domain, port):
        """Start one frozen teacher. It only prefills student responses to score them; it never generates."""
        gpus = recipe.TEACHER_PLACEMENT[domain][1]
        self.processes.launch([
            sys.executable, '-m', 'sglang.launch_server', '--model-path', ROOT / 'teachers' / domain,
            '--tokenizer-path', BASE_MODEL, '--host', '0.0.0.0', '--port', port, '--tp-size', len(gpus),
            '--context-length', recipe.CONTEXT_LENGTH, '--mem-fraction-static', '0.8',
            '--max-running-requests', '128', '--disable-cuda-graph', '--chunked-prefill-size', '-1',
            '--disable-radix-cache', '--prefill-max-requests', '1', '--moe-runner-backend', 'triton',
            '--enable-metrics'], f'teacher-{domain}', {'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus))})

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

    def verify_placement(self, generation_ips):
        """Check that the framework places the trainer and policy on the intended physical GPUs."""
        import ray
        from ray.util import remove_placement_group

        total = recipe.TRAINER_GPUS + recipe.ROLLOUT_GPUS
        ray.init(address=f'{self.ip}:{RAY_PORT}')
        deadline = time.monotonic() + 180
        while int(ray.cluster_resources().get('GPU', 0)) != total:
            self.processes.check()
            assert time.monotonic() < deadline, ray.cluster_resources()
            time.sleep(2)
        nodes = {n['NodeManagerAddress']: n['Resources'].get('GPU') for n in ray.nodes() if n['Alive']}
        assert nodes == {self.ip: float(recipe.TRAINER_GPUS),
                         **{ip: float(len(recipe.POLICY_GPUS)) for ip in generation_ips}}, nodes
        # The frameworks sort bundles by node IP, so the trainer must have the lowest address.
        assert all(ipaddress.ip_address(self.ip) < ipaddress.ip_address(ip) for ip in generation_ips), nodes
        gpu_ids, group = self.placement(importlib.import_module(self.placement_module)._create_placement_group(total))
        try:
            assert gpu_ids == list(range(recipe.TRAINER_GPUS)) + list(recipe.POLICY_GPUS) * len(generation_ips), gpu_ids
            write_json(self.result / 'placement-verified.json', {'nodes': nodes, 'sorted_physical_gpu_ids': gpu_ids})
        finally:
            remove_placement_group(group)
            ray.shutdown()
