"""The Slime recipe: native `train_async.py` arguments built from `shared.recipe`.

Run by the learner role. The argument list is retained as `argv.json`.
"""
import json
import os
import subprocess
import sys

from shared import recipe
from shared.slurm import DATA, ROOT, eval_config, result_dir, site

# Environment that Ray workers need for the hooks, logging and process groups.
RAY_WORKER_ENV = ('CAMPAIGN_RESULT', 'CAMPAIGN_ROOT', 'CAMPAIGN_TEACHER_URLS', 'CAMPAIGN_PYDEPS', 'PYTHONPATH', 'TENSORBOARD_DIR',
                  'NCCL_DEBUG', 'CUDA_DEVICE_MAX_CONNECTIONS', 'SLIME_NATIVE_PROCESS_GROUPS')


def model_args():
    """`MODEL_ARGS` from Slime's own script for the Qwen3.5/3.6 MoE architecture."""
    script = ROOT / 'source/scripts/models/qwen3.5-35B-A3B.sh'
    output = subprocess.check_output(['bash', '-c', 'source "$1"; printf "%s\\0" "${MODEL_ARGS[@]}"', 'bash', script])
    return output.decode().rstrip('\0').split('\0')


def build(result):
    s, r = site(), recipe
    (result / 'eval.json').write_text(json.dumps(eval_config(), indent=2))
    args = [*model_args(),
            '--hf-checkpoint', s['base_model'], '--ref-load', s['megatron_model'],
            '--save', ROOT / 'checkpoints' / result.name, '--save-interval', r.SAVE_INTERVAL,
            '--no-save-optim', '--no-save-rng',
            # Hugging Face weights of each saved update, for benchmarking without a conversion.
            '--save-hf', ROOT / 'checkpoints' / result.name / 'hf' / 'iter_{rollout_id}',
            # Data and rollout.
            '--prompt-data', DATA / r.TRAIN_FILE, '--input-key', 'prompt', '--label-key', 'label',
            '--metadata-key', 'metadata', '--apply-chat-template', '--apply-chat-template-kwargs',
            json.dumps({'enable_thinking': r.ENABLE_THINKING}, separators=(',', ':')),
            '--rollout-skip-special-tokens', '--rollout-shuffle',
            '--num-rollout', r.UPDATES, '--rollout-batch-size', r.PROMPTS_PER_UPDATE,
            '--n-samples-per-prompt', r.SAMPLES_PER_PROMPT, '--global-batch-size', r.PROMPTS_PER_UPDATE,
            '--rollout-max-response-len', r.MAX_RESPONSE_TOKENS, '--rollout-temperature', '1', '--rollout-top-p', '1',
            # opd_hooks.py routes each domain to its teacher; task scores are diagnostics only.
            '--custom-rm-path', 'opd_hooks.reward', '--custom-reward-post-process-path', 'opd_hooks.postprocess',
            '--reward-key', 'task_score', '--eval-reward-key', 'task_score',
            '--rollout-sample-hook-path', 'opd_hooks.stamp', '--custom-rollout-log-function-path', 'opd_hooks.log_train',
            '--custom-eval-rollout-log-function-path', 'opd_hooks.log_eval',
            # Native OPD objective with mismatch metrics only (no TIS or rejection sampling).
            '--use-opd', '--opd-type', 'sglang', '--opd-kl-coef', '1.0', '--advantage-estimator', 'grpo',
            '--eps-clip', '0.2', '--eps-clip-high', '0.2', '--entropy-coef', '0', '--kl-coef', '0', '--kl-loss-coef', '0',
            '--get-mismatch-metrics', '--custom-config-path', ROOT / 'mismatch-metrics.yaml',
            '--custom-tis-function-path', 'examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp',
            # Optimizer.
            '--optimizer', 'adam', '--lr', r.LEARNING_RATE, '--lr-decay-style', 'constant', '--weight-decay', '0',
            '--adam-beta1', '0.9', '--adam-beta2', '0.999', '--clip-grad', '1', '--bf16', '--use-precision-aware-optimizer',
            # Trainer parallelism: expert parallel over all trainer GPUs, no context parallelism.
            '--tensor-model-parallel-size', '1', '--pipeline-model-parallel-size', '1', '--context-parallel-size', '1',
            '--expert-model-parallel-size', r.TRAINER_GPUS, '--expert-tensor-parallel-size', '1',
            '--recompute-granularity', 'full', '--recompute-method', 'uniform', '--recompute-num-layers', '1',
            '--use-dynamic-batch-size', '--max-tokens-per-gpu', r.MAX_TOKENS_PER_GPU, '--balance-data',
            '--log-probs-chunk-size', r.LOG_PROBS_CHUNK_SIZE,
            '--attention-dropout', '0', '--hidden-dropout', '0', '--accumulate-allreduce-grads-in-fp32',
            '--attention-softmax-in-fp32', '--attention-backend', 'flash',
            # Placement and SGLang policy engines.
            '--actor-num-nodes', '1', '--actor-num-gpus-per-node', r.TRAINER_GPUS, '--num-gpus-per-node', r.GPUS_PER_NODE,
            '--rollout-num-gpus', len(r.POLICY_GPUS), '--rollout-num-gpus-per-engine', r.POLICY_GPUS_PER_ENGINE,
            '--sglang-ep-size', r.POLICY_GPUS_PER_ENGINE, '--sglang-moe-runner-backend', 'triton',
            '--sglang-mem-fraction-static', '0.65', '--sglang-max-running-requests', '256',
            '--sglang-server-concurrency', r.PROMPTS_PER_UPDATE, '--sglang-context-length', r.CONTEXT_LENGTH,
            '--sglang-mamba-scheduler-strategy', 'extra_buffer', '--sglang-enable-metrics',
            '--no-offload-train', '--no-offload-rollout',
            # Evaluation, weight sync after every update, and logging.
            '--eval-interval', r.EVAL_INTERVAL, '--eval-config', result / 'eval.json', '--eval-temperature', '0',
            '--n-samples-per-eval-prompt', '1', '--eval-max-response-len', r.MAX_RESPONSE_TOKENS,
            '--update-weights-interval', '1', '--use-tensorboard', '--tb-project-name', result / 'tensorboard',
            '--tb-experiment-name', 'slime-mopd',
            '--save-debug-rollout-data', result / 'native-rollouts' / 'rollout_{rollout_id}.pt']
    args = [str(a) for a in args]
    (result / 'argv.json').write_text(json.dumps(args, indent=2))
    return args


if __name__ == '__main__':
    from slime.utils.arguments import parse_args
    from train_async import train
    import ray

    result = result_dir()
    sys.argv = ['train_async.py', *build(result)]
    parsed = parse_args()
    (result / 'resolved-args.json').write_text(json.dumps(vars(parsed), indent=2, default=str))
    ray.init(address=os.environ['RAY_ADDRESS'], runtime_env={'env_vars': {k: os.environ[k] for k in RAY_WORKER_ENV}})
    train(parsed)
