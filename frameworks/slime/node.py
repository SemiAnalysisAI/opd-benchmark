"""Slime entry point: `node.py` runs the controller, `node.py ROLE` runs one container role.

The two-node runtime lives in `shared/container.py`. This file adds Slime's pinned wheels,
placement format and training driver (`train.py`).
"""
import json
import subprocess
import sys

from shared.container import RAY_PORT, ContainerNode, entry
from shared.slurm import ROOT


class SlimeNode(ContainerNode):
    placement_module = 'slime.ray.placement_group'
    # Native NCCL groups are safe because trainer offload and release are disabled.
    env = {'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'SLIME_NATIVE_PROCESS_GROUPS': '1'}

    def setup(self):
        # Install the pinned wheels into this throwaway container only; the image stays unchanged.
        with (self.result / f'{self.role}-numpy-install.out').open('wb') as log:
            subprocess.run(['sha256sum', '-c', 'wheels.sha256'], cwd=ROOT, stdout=log, stderr=log, check=True)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', '--no-index',
                            '--find-links', str(ROOT / 'wheels'), 'numpy==1.26.4', 'scipy==1.15.3'],
                           stdout=log, stderr=log, check=True, timeout=120)

    def placement(self, group):
        # Slime returns (placement group, bundle indices, sorted physical GPU ids).
        return [int(x) for x in group[2]], group[0]

    def train_command(self, teacher_urls):
        return [sys.executable, str(ROOT / 'train.py')], {
            'RAY_ADDRESS': f'{self.ip}:{RAY_PORT}', 'CAMPAIGN_TEACHER_URLS': json.dumps(teacher_urls)}


if __name__ == '__main__':
    sys.exit(entry('slime', SlimeNode))
