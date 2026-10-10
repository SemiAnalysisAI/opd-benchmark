"""NeMo-RL recipe: MOPD on async GRPO with NeMo Gym rollouts, built from `shared.recipe`.

Usage: `make_config.py RESULT_DIR` (called by `node.py`). Writes `nemo-gym-data/{train,validation}.jsonl`
and `nemo-rl.yaml` (JSON, which is valid YAML) into the result directory.

The config extends upstream's base GRPO config. Algorithm settings come from the MOPD reference recipe,
`examples/configs/recipes/llm/mopd-qwen3-1.7b-3n8g-megatron-pack.yaml`; Qwen3.5 settings come from
`grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2cp2.yaml`, except its eager vLLM (see `vllm_cfg`). MOPD gives the
teacher and vLLM generation a whole node each, so there are three nodes: trainer, vLLM and teacher.
"""
import json
from pathlib import Path
import sys

from shared import recipe
from shared.slurm import DATA, ROOT, site

UPSTREAM = Path('/opt/nemo-rl')  # The NGC container's copy of the pinned release.
AGENT = 'reasoning_gym_simple_agent'  # NeMo Gym's reasoning_gym agent (resources_servers/reasoning_gym).


def gym_rows(name):
    """Convert packaged rows to NeMo Gym's reasoning_gym format, keeping the chat messages as is."""
    for line in (DATA / name).read_text().splitlines():
        row = json.loads(line)
        entry = json.loads(row['label'])
        yield json.dumps({'responses_create_params': {'input': row['prompt']}, 'question': entry['question'],
                          'answer': entry['answer'], 'metadata': entry['metadata'],
                          'agent_ref': {'type': 'responses_api_agents', 'name': AGENT}})


def recipe_input(result, data, s):
    r = recipe
    return {
        'defaults': str(UPSTREAM / 'examples/configs/grpo_math_1B.yaml'),
        'grpo': {
            'num_prompts_per_step': r.PROMPTS_PER_UPDATE, 'num_generations_per_prompt': r.SAMPLES_PER_PROMPT,
            'max_num_steps': r.UPDATES, 'val_period': r.EVAL_INTERVAL, 'val_at_start': True,
            'max_val_samples': None, 'val_batch_size': r.EVAL_EXAMPLES[r.DOMAINS[0]], 'overlong_filtering': True,
            'async_grpo': {'enabled': True, 'max_trajectory_age_steps': r.MAX_POLICY_LAG},
            'adv_estimator': {'name': 'opd'}, 'seq_logprob_error_threshold': 2.0,
        },
        # As in the MOPD reference recipe: REINFORCE form with an ICE-POP importance-sampling gate.
        'loss_fn': {
            'reference_policy_kl_penalty': 0.0, 'kl_input_clamp_value': None, 'kl_output_clamp_value': None,
            'ratio_clip_max': 0.28, 'use_on_policy_kl_approximation': True, 'disable_ppo_ratio': True,
            'use_importance_sampling_correction': True, 'truncated_importance_sampling_ratio': 5.0,
            'truncated_importance_sampling_ratio_min': 0.2, 'truncated_importance_sampling_type': 'icepop',
        },
        'checkpointing': {
            'enabled': True, 'checkpoint_dir': str(ROOT / 'checkpoints' / result.name), 'save_period': r.SAVE_INTERVAL,
            'keep_top_k': 1, 'metric_name': None, 'save_optimizer': False,
        },
        'policy': {
            'model_name': s['base_model'], 'tokenizer': {'name': s['base_model'],
                                                          'chat_template_kwargs': {'enable_thinking': r.ENABLE_THINKING}},
            'train_global_batch_size': r.PROMPTS_PER_UPDATE * r.SAMPLES_PER_PROMPT, 'train_micro_batch_size': 1,
            'logprob_batch_size': 1, 'max_total_sequence_length': r.CONTEXT_LENGTH,
            'logprob_chunk_size': r.LOG_PROBS_CHUNK_SIZE, 'max_grad_norm': 1.0, 'make_sequence_length_divisible_by': 8,
            'dtensor_cfg': {'enabled': False},
            'megatron_cfg': {
                'enabled': True, 'tensor_model_parallel_size': 1, 'pipeline_model_parallel_size': 1,
                'context_parallel_size': 1, 'expert_model_parallel_size': r.TRAINER_GPUS, 'sequence_parallel': False,
                'activation_checkpointing': True, 'bias_activation_fusion': False, 'apply_rope_fusion': False,
                'defer_fp32_logits': True,
                'optimizer': {'lr': r.LEARNING_RATE, 'min_lr': r.LEARNING_RATE, 'weight_decay': 0.0,
                              'adam_beta1': 0.9, 'adam_beta2': 0.999},
                'scheduler': {'lr_decay_style': 'constant', 'lr_decay_iters': None, 'lr_warmup_iters': 0,
                              'lr_warmup_init': r.LEARNING_RATE},
                'distributed_data_parallel_config': {'average_in_collective': False},
            },
            'optimizer': None, 'scheduler': None,
            'generation': {
                'max_new_tokens': r.MAX_RESPONSE_TOKENS, 'temperature': 1.0, 'top_p': 1.0,
                'vllm_cfg': {
                    # TP1: with TP2 the first trainer-to-vLLM weight sync deadlocked twice, with the receive
                    # never reaching vLLM's Ray TP executor. TP1 avoids that executor, though the sync still hung (see repro/README.md).
                    'async_engine': True, 'tensor_parallel_size': 1,
                    # CUDA graphs stay on (the base config's default). Upstream's Qwen3.5 recipes set enforce_eager,
                    # but they generate at most 4k tokens; at 30k thinking tokens eager decoding took ~21 min per rollout.
                    'gpu_memory_utilization': 0.8, 'max_model_len': r.CONTEXT_LENGTH, 'enforce_eager': False,
                    'expose_http_server': True,
                    'http_server_serving_chat_kwargs': {'enable_auto_tools': True, 'tool_parser': 'hermes',
                                                        'reasoning_parser': 'qwen3'},
                },
                # With more than one policy node, vLLM takes a whole node (grpo.py requires it).
                'colocated': {'enabled': False, 'resources': {'gpus_per_node': r.GPUS_PER_NODE, 'num_nodes': 1}},
            },
        },
        'data': {
            'max_input_seq_length': None,
            'train': {'data_path': str(data / 'train.jsonl'), 'dataset_name': 'NemoGymDataset'},
            'validation': {'data_path': str(data / 'validation.jsonl'), 'dataset_name': 'NemoGymDataset'},
            'default': {'dataset_name': 'NemoGymDataset', 'env_name': 'nemo_gym', 'prompt_file': None,
                        'processor': 'nemo_gym_data_processor'},
        },
        'env': {'should_use_nemo_gym': True, 'nemo_gym': {'config_paths': [
            'responses_api_models/vllm_model/configs/vllm_model_for_training.yaml',
            'resources_servers/reasoning_gym/configs/reasoning_gym.yaml']}},
        'logger': {'log_dir': str(result / 'logs'), 'tensorboard_enabled': True, 'wandb_enabled': False,
                   'mlflow_enabled': False, 'monitor_gpus': True},
        'cluster': {'gpus_per_node': r.GPUS_PER_NODE, 'num_nodes': 3},
        'on_policy_distillation': {
            'enabled': True, 'teacher_model_by_agent_name': {'default_teacher': f'{s["teachers"]}/{r.DOMAINS[0]}'},
            'default_teacher_alias': 'default_teacher', 'strict_agent_name_match': False,
            'deduplicate_shared_teacher_checkpoints': True,
            'non_colocated_teachers': {'enabled': True, 'default_teacher_cfg': {
                # One GPU, as in the other frameworks; MOPD still reserves the whole node.
                'tensor_model_parallel_size': 1, 'pipeline_model_parallel_size': 1, 'context_parallel_size': 1,
                'num_nodes': 1, 'gpus_per_node': 1, 'precision': 'bf16', 'micro_batch_size': 1}},
        },
    }


def main(result):
    assert len(recipe.DOMAINS) == 1, 'The NeMo-RL recipe routes every sample to one teacher'
    data = result / 'nemo-gym-data'
    data.mkdir(exist_ok=True)
    (data / 'train.jsonl').write_text('\n'.join(gym_rows(recipe.TRAIN_FILE)) + '\n')
    (data / 'validation.jsonl').write_text('\n'.join(gym_rows(recipe.data_file(recipe.DOMAINS[0], 'dev'))) + '\n')
    (result / 'nemo-rl.yaml').write_text(json.dumps(recipe_input(result, data, site()), indent=2) + '\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
