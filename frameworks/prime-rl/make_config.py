"""The Prime-RL recipe and its component configs for the two-node layout.

Usage: `make_config.py RESULT_DIR` (run by `node.py`). `recipe_input` is the
experiment in Prime-RL's own schema; `main` validates it and writes one config
per component that `node.py` launches.
"""

import json
from pathlib import Path
import sys

from shared import recipe
from shared.slurm import DATA, ROOT, site

# Generation node: three independent TP2 vLLM policy engines behind a router,
# and one single-GPU vLLM server per frozen teacher.
ROUTER_PORT = 8000
ROUTER_METRICS_PORT = 29000
POLICY_PORTS = (8100, 8110, 8120)
TEACHER_PORTS = {domain: 8200 + 100 * i for i, domain in enumerate(recipe.DOMAINS)}
RPC_PORT_OFFSET = 1000  # Each vLLM server's data-parallel RPC port is its HTTP port + 1000.
INFERENCE_PORTS = (*POLICY_PORTS, *TEACHER_PORTS.values())
GENERATION_NODE_PORTS = (ROUTER_PORT, *INFERENCE_PORTS, *(p + RPC_PORT_OFFSET for p in INFERENCE_PORTS),
                         ROUTER_METRICS_PORT)
# Trainer node.
WEIGHT_BROADCAST_PORT = 29591
ROLLOUT_TRANSPORT_PORT = 15555
TRAINER_NODE_PORTS = (5000, 5001, 5002, 5003, ROLLOUT_TRANSPORT_PORT, ROLLOUT_TRANSPORT_PORT + 1,
                      WEIGHT_BROADCAST_PORT)


def policy_gpus(index):
    """Physical GPUs of policy engine `index` on the generation node."""
    size = recipe.POLICY_GPUS_PER_ENGINE
    return recipe.POLICY_GPUS[index * size:(index + 1) * size]


def task_source(domain, split, teachers, generation_ip):
    value = {
        'name': domain,
        'env': {
            'taskset': {'id': 'rg-tasks', 'data_path': str(DATA / recipe.data_file(domain, split)), 'domain': domain},
            'agent': {'harness': {'id': 'null'}, 'runtime': {'type': 'subprocess'}},
        },
    }
    if split == 'train':
        # Native per-source OPD: each domain is distilled from its own frozen teacher.
        value['ratio'] = 1.0
        value['algo'] = {'type': 'opd', 'teacher': {
            'name': f'{teachers}/{domain}',
            'base_url': f'http://{generation_ip}:{TEACHER_PORTS[domain]}/v1'}}
    return value


def recipe_input(result, s):
    """The Prime-RL recipe. The resolver sizes components from the logical GPU budget below.

    Everything not set here is Prime-RL's default. Set are only the shared recipe (updates, lengths, batch,
    learning rate, weight decay 0, policy lag, thinking, evaluation), the two-node layout (GPU budget, ports,
    NCCL weight broadcast, three TP2 policy engines) and the per-domain OPD teachers. The student is trained as
    a text model (no `model.vlm`, as README.md lists Qwen3.5 MoE), so the trainer keeps its default fp32
    master weights and reductions; `model.vlm` would require bf16 for both. The frozen vision encoder then runs in
    fp32, which FlashAttention 4 rejects; the package's source patch gives it SDPA (see upstream/manifest.json).
    """
    generation_ip = s['generation_ip']
    sources = {split: [task_source(d, split, s['teachers'], generation_ip) for d in recipe.DOMAINS]
               for split in ('train', 'dev')}
    return {
        'max_steps': recipe.UPDATES, 'seq_len': recipe.CONTEXT_LENGTH, 'output_dir': str(result.parent),
        'run': {'name': result.name}, 'dashboard': False,
        'model': {'name': s['base_model']},
        'deployment': {'type': 'single_node', 'gpus_per_node': 2 * recipe.GPUS_PER_NODE,
                       'num_train_gpus': recipe.TRAINER_GPUS, 'num_infer_gpus': len(recipe.POLICY_GPUS)},
        'weight_broadcast': {'type': 'nccl', 'host': s['trainer_ip'], 'port': WEIGHT_BROADCAST_PORT},
        'rollout_transport': {'type': 'zmq', 'host': '127.0.0.1', 'port': ROLLOUT_TRANSPORT_PORT},
        'monitors': {'file': {'chunk_bytes': 16777216, 'compress': True}},
        'ckpt': {'interval': recipe.SAVE_INTERVAL, 'output_dir': str(ROOT / 'checkpoints' / result.name)},
        'trainer': {'optim': {'lr': recipe.LEARNING_RATE, 'weight_decay': 0.0}},
        'orchestrator': {
            'batch_size': recipe.PROMPTS_PER_UPDATE, 'group_size': recipe.SAMPLES_PER_PROMPT,
            'max_off_policy_steps': recipe.MAX_POLICY_LAG,
            'model': {'client': {'base_url': f'http://{generation_ip}:{ROUTER_PORT}/v1',
                                 'admin_base_url': [f'http://{generation_ip}:{port}/v1' for port in POLICY_PORTS]}},
            'renderer': {'name': 'qwen3.6', 'enable_thinking': recipe.ENABLE_THINKING},
            'train': {'source': sources['train'],
                      'sampling': {'temperature': 1.0, 'top_p': 1.0, 'max_completion_tokens': recipe.MAX_RESPONSE_TOKENS}},
            # num_examples -1 evaluates every dev row.
            'eval': {'interval': recipe.EVAL_INTERVAL, 'num_examples': -1, 'group_size': 1,
                     'sampling': {'temperature': 0.0, 'max_completion_tokens': recipe.MAX_RESPONSE_TOKENS,
                                  'extra_body': {'chat_template_kwargs': {'enable_thinking': recipe.ENABLE_THINKING}}},
                     'source': sources['dev']},
        },
        'inference': {
            'server': {'host': '0.0.0.0', 'port': ROUTER_PORT}, 'backend_port': POLICY_PORTS[0],
            'vllm': {'tensor_parallel_size': recipe.POLICY_GPUS_PER_ENGINE, 'data_parallel_size': len(POLICY_PORTS),
                     'data_parallel_size_local': len(POLICY_PORTS), 'api_server_count': len(POLICY_PORTS),
                     'max_model_len': recipe.CONTEXT_LENGTH},
        },
    }


def teacher_input(domain, result, s):
    port = TEACHER_PORTS[domain]
    return {
        'server': {'host': '0.0.0.0', 'port': port},
        'router': None,
        'output_dir': str(result / f'teacher-{domain}'),
        'vllm': {'model': f'{s["teachers"]}/{domain}', 'tensor_parallel_size': 1,
                 'data_parallel_rpc_port': port + RPC_PORT_OFFSET, 'max_model_len': recipe.CONTEXT_LENGTH,
                 # As upstream's OPD example starts its teacher (configs/debug/algo/opd.toml). At the default 0.9 the
                 # teacher ran out of memory: the fp32 prompt logprobs of one 9.8k-token sample need 8.8 GiB
                 # outside vLLM's profiled budget (job 1016).
                 'gpu_memory_utilization': 0.5, 'enforce_eager': True},
    }


def main(result):
    from prime_rl.configs.inference import InferenceConfig
    from prime_rl.configs.rl import RLConfig
    from prime_rl.entrypoints.rl import write_subconfigs
    from prime_rl.utils.config import dump_resolved_config
    from prime_rl.utils.pathing import create_attempt_dirs

    s = site()
    raw = recipe_input(result, s)
    # Written before validation, which replaces nested env dicts in `raw` with config objects.
    (result / 'recipe-input.json').write_text(json.dumps(raw, indent=2))
    config = RLConfig.model_validate(raw)
    config_dir, log_dir = create_attempt_dirs(result)
    (config_dir / 'rl.json').write_text(json.dumps(dump_resolved_config(config), indent=2))
    write_subconfigs(config, config_dir)

    # The aggregate resolver represents six workers. Each executable engine is an
    # independent TP2/DP1 group because vLLM folds MoE DP into TP when EP is off.
    replicas = []
    for index, port in enumerate(POLICY_PORTS):
        replica_raw = dump_resolved_config(config.inference)
        replica_raw.update(router=None, server={'host': '0.0.0.0', 'port': port},
                           output_dir=str(result / f'policy-{index}'))
        replica_raw['vllm'].update(data_parallel_size=1, data_parallel_size_local=1, api_server_count=1,
                                   data_parallel_rpc_port=port + RPC_PORT_OFFSET)
        replica = InferenceConfig.model_validate(replica_raw)
        assert replica.vllm.tensor_parallel_size == recipe.POLICY_GPUS_PER_ENGINE
        assert replica.vllm.data_parallel_size == 1
        (config_dir / f'policy-{index}.json').write_text(json.dumps(dump_resolved_config(replica), indent=2))
        replicas.append({'name': f'policy-{index}', 'port': port, 'gpus': list(policy_gpus(index)),
                         'tp': recipe.POLICY_GPUS_PER_ENGINE, 'dp': 1})
    (result / 'policy-json').write_text(json.dumps({
        'aggregate_inference_config_is_resolver_only': True, 'router_port': ROUTER_PORT,
        'replicas': replicas, 'broadcast_world_size': len(recipe.POLICY_GPUS),
        'broadcast_rank_offsets': [i * recipe.POLICY_GPUS_PER_ENGINE for i in range(len(replicas))]}, indent=2))

    for domain in recipe.DOMAINS:
        teacher = InferenceConfig.model_validate(teacher_input(domain, result, s))
        (config_dir / f'teacher-{domain}.json').write_text(json.dumps(dump_resolved_config(teacher), indent=2))

    assert config.trainer.weight_broadcast.inference_world_size == len(recipe.POLICY_GPUS)
    assert config.trainer.rollout_transport.port == config.orchestrator.rollout_transport.port == ROLLOUT_TRANSPORT_PORT
    assert config.orchestrator.num_train_workers == recipe.TRAINER_GPUS
    assert all(source.algo.type == 'opd' for source in config.orchestrator.train.source)
    (result / 'resolved-paths.json').write_text(json.dumps({'config': str(config_dir), 'logs': str(log_dir)}))
    print(json.dumps({'config': str(config_dir), 'logs': str(log_dir), 'native_opd_sources': len(recipe.DOMAINS)}))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
