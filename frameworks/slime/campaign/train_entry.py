"""Validate native arguments before invoking the instrumented native driver."""

import json
import os
from pathlib import Path
import sys

from make_args import build
from slime.utils.arguments import parse_args
from train_async import train

result = Path(os.environ['MOPD_RESULT'])
sys.argv = ['train_async.py', *build(result)]
args = parse_args()
assert args.num_rollout == 40 and args.global_batch_size == 128 and args.use_opd
assert args.actor_num_gpus_per_node == 8 and args.rollout_num_gpus == 6 and not args.colocate
assert args.update_weights_interval == 1
assert not args.use_tis and not args.use_rs
assert not getattr(args, 'profile', False) and not getattr(args, 'record_memory_history', False)
(result/'resolved-args.json').write_text(json.dumps(vars(args), indent=2, default=str))
if os.environ.get('MOPD_PARSE_ONLY') != '1':
    import ray
    inherited = {key: os.environ[key] for key in (
        'MOPD_RESULT', 'MOPD_CAMPAIGN', 'MOPD_TEACHER_URLS', 'PYTHONPATH',
        'TENSORBOARD_DIR', 'NCCL_DEBUG', 'CUDA_DEVICE_MAX_CONNECTIONS', 'SLIME_NATIVE_PROCESS_GROUPS')}
    ray.init(address=os.environ['RAY_ADDRESS'], runtime_env={'env_vars': inherited})
    train(args)
