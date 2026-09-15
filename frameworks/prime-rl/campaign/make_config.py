"""Resolve native Prime-RL component configs for the allocated two-node layout."""

import json
import os
from pathlib import Path
import sys

from prime_rl.configs.rl import RLConfig
from prime_rl.configs.inference import InferenceConfig
from prime_rl.entrypoints.rl import write_subconfigs
from prime_rl.utils.config import dump_resolved_config
from prime_rl.utils.pathing import create_attempt_dirs

ROOT = Path(__file__).resolve().parent
BASE = '@BASE_MODEL@'
TEACHERS = '@TEACHERS@'
result, learner_ip, generation_ip = sys.argv[1:]
result = Path(result)


def source(domain, split):
    filename = ('countdown4' if domain == 'countdown' else 'graph12') + f'-{split}.jsonl'
    value = {
        'name': domain,
        'env': {
            'taskset': {'id': 'mopd-puzzles', 'data_path': str(ROOT / 'data' / filename), 'domain': domain},
            'agent': {'harness': {'id': 'null'}, 'runtime': {'type': 'subprocess'}},
        },
    }
    if split == 'train':
        port = 8200 if domain == 'countdown' else 8300
        value['ratio'] = 1.0
        value['algo'] = {'type': 'opd', 'teacher': {'name': f'{TEACHERS}/{domain}',
                         'base_url': f'http://{generation_ip}:{port}/v1'}}
    return value


# The shared resolver derives component sizes from the logical GPU budget.
# The controller launches these native components on two physical nodes.
raw = {
    'max_steps': 40, 'seq_len': 2048, 'output_dir': str(result.parent),
    'run': {'name': result.name}, 'dashboard': False,
    'model': {'name': BASE, 'vlm': {'vision_encoder_attr': 'model.visual',
              'language_model_attr': 'model.language_model', 'freeze_vision_encoder': True}},
    'deployment': {'type': 'single_node', 'gpus_per_node': 16, 'num_train_gpus': 8, 'num_infer_gpus': 6},
    'weight_broadcast': {'type': 'nccl', 'host': learner_ip, 'port': 29591, 'timeout': 1200},
    'rollout_transport': {'type': 'zmq', 'host': '127.0.0.1', 'port': 15555},
    'monitors': {'file': {'chunk_bytes': 16777216, 'compress': True}},
    'ckpt': {'interval': 20, 'keep_last': 1, 'output_dir': str(ROOT / 'checkpoints' / result.name)},
    'trainer': {
        'model': {'impl': 'custom', 'ep': 8, 'optim_cpu_offload': False,
                  'optimization_dtype': 'bfloat16', 'reduce_dtype': 'bfloat16',
                  'ac_offloading': None, 'conversion_dir': BASE},
        'optim': {'lr': 1e-6, 'weight_decay': 0.0},
    },
    'orchestrator': {
        'batch_size': 128, 'group_size': 1, 'max_off_policy_steps': 2,
        'concurrency': {'initial_inflight': 128, 'min_inflight': 32, 'max_inflight': 256},
        'model': {'client': {'base_url': f'http://{generation_ip}:8000/v1',
                            'admin_base_url': [f'http://{generation_ip}:{port}/v1' for port in (8100, 8110, 8120)]}},
        'renderer': {'name': 'qwen3.6', 'enable_thinking': False},
        'train': {'source': [source(d, 'train') for d in ('countdown', 'graph_color')],
                  'sampling': {'temperature': 1.0, 'top_p': 1.0, 'max_completion_tokens': 256}},
        'eval': {'interval': 10, 'num_examples': 512, 'group_size': 1,
                 'sampling': {'temperature': 0.0, 'max_completion_tokens': 256,
                              'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}},
                 'source': [source(d, 'dev') for d in ('countdown', 'graph_color')]},
    },
    'inference': {
        'server': {'host': '0.0.0.0', 'port': 8000}, 'backend_port': 8100,
        'vllm': {'tensor_parallel_size': 2, 'data_parallel_size': 3,
                 'data_parallel_size_local': 3, 'api_server_count': 3,
                 'gpu_memory_utilization': 0.8, 'max_model_len': 2048,
                 'max_num_seqs': 128, 'max_num_batched_tokens': 8192,
                 'enable_expert_parallel': False, 'dtype': 'bfloat16'},
    },
}
raw_json = json.dumps(raw, indent=2)
config = RLConfig.model_validate(raw)
config_dir, log_dir = create_attempt_dirs(result)
(result / 'recipe-input.json').write_text(raw_json)
(config_dir / 'rl.json').write_text(json.dumps(dump_resolved_config(config), indent=2))
write_subconfigs(config, config_dir)
# The aggregate resolver represents six workers. Each executable engine is an
# independent TP2/DP1 group because vLLM folds MoE DP into TP when EP is off.
policy_replicas = []
for index, port in enumerate((8100, 8110, 8120)):
    replica_raw = dump_resolved_config(config.inference)
    replica_raw.update(router=None, server={'host': '0.0.0.0', 'port': port},
                       output_dir=str(result / f'policy-{index}'))
    replica_raw['vllm'].update(data_parallel_size=1, data_parallel_size_local=1,
                               api_server_count=1, data_parallel_rpc_port=port + 1000)
    replica = InferenceConfig.model_validate(replica_raw)
    assert replica.vllm.tensor_parallel_size == 2
    assert replica.vllm.data_parallel_size == 1
    (config_dir / f'policy-{index}.json').write_text(json.dumps(dump_resolved_config(replica), indent=2))
    policy_replicas.append({'name': f'policy-{index}', 'port': port,
                            'gpus': [2 + index * 2, 3 + index * 2], 'tp': 2, 'dp': 1})
(result / 'policy-topology.json').write_text(json.dumps({
    'aggregate_inference_config_is_resolver_only': True,
    'router_port': 8000, 'replicas': policy_replicas,
    'broadcast_world_size': 6, 'broadcast_rank_offsets': [0, 2, 4]}, indent=2))
for domain, port in [('countdown', 8200), ('graph_color', 8300)]:
    teacher = InferenceConfig.model_validate({
        'server': {'host': '0.0.0.0', 'port': port},
        'router': None,
        'output_dir': str(result / f'teacher-{domain}'),
        'vllm': {'model': f'{TEACHERS}/{domain}', 'dtype': 'bfloat16',
                 'tensor_parallel_size': 1, 'data_parallel_rpc_port': port + 1000,
                 'gpu_memory_utilization': 0.8, 'max_model_len': 2048,
                 'max_num_seqs': 128, 'max_num_batched_tokens': 4096,
                 'enable_prefix_caching': False, 'enforce_eager': True, 'language_model_only': True},
    })
    (config_dir / f'teacher-{domain}.json').write_text(json.dumps(dump_resolved_config(teacher), indent=2))
assert config.trainer.weight_broadcast.inference_world_size == 6
assert config.trainer.rollout_transport.port == config.orchestrator.rollout_transport.port == 15555
assert config.orchestrator.num_train_workers == 8
assert all(s.algo.type == 'opd' for s in config.orchestrator.train.source)
(result / 'resolved-paths.json').write_text(json.dumps({'config': str(config_dir), 'logs': str(log_dir)}))
print(json.dumps({'config': str(config_dir), 'logs': str(log_dir), 'native_opd_sources': 2}))
