"""Build the exact native Slime recipe and retain all effective arguments."""

import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = '@BASE_MODEL@'
CONVERTED = '@MEGATRON_MODEL@'


def build(result):
    model = subprocess.check_output(['bash', '-c', 'source "$1"; printf "%s\\0" "${MODEL_ARGS[@]}"', 'bash', str(ROOT/'source/scripts/models/qwen3.5-35B-A3B.sh')]).decode().rstrip('\0').split('\0')
    eval_config = {'eval': {'defaults': {'temperature': 0, 'n_samples_per_eval_prompt': 1,
        'max_response_len': 256}, 'datasets': [
        {'name': domain, 'path': str(ROOT/'data'/f'{prefix}-dev.jsonl')}
        for domain, prefix in [('countdown','countdown4'), ('graph_color','graph12')]]}}
    (result/'eval.json').write_text(json.dumps(eval_config, indent=2))
    args = model + [
        '--hf-checkpoint', BASE, '--ref-load', CONVERTED,
        '--save', str(ROOT/'checkpoints'/result.name), '--save-interval', '20', '--no-save-optim', '--no-save-rng',
        '--prompt-data', str(ROOT/'data/mixed-train.jsonl'), '--input-key', 'prompt', '--label-key', 'label',
        '--metadata-key', 'metadata', '--apply-chat-template', '--apply-chat-template-kwargs', '{"enable_thinking":false}',
        '--rollout-skip-special-tokens', '--rollout-shuffle', '--num-rollout', '40', '--rollout-batch-size', '128',
        '--n-samples-per-prompt', '1', '--global-batch-size', '128', '--rollout-max-response-len', '256',
        '--rollout-temperature', '1', '--rollout-top-p', '1',
        '--custom-rm-path', 'campaign.reward', '--custom-reward-post-process-path', 'campaign.postprocess',
        '--reward-key', 'task_score', '--eval-reward-key', 'task_score',
        '--rollout-sample-hook-path', 'campaign.stamp', '--custom-rollout-log-function-path', 'campaign.log_train',
        '--custom-eval-rollout-log-function-path', 'campaign.log_eval',
        '--use-opd', '--opd-type', 'sglang', '--opd-kl-coef', '1.0', '--advantage-estimator', 'grpo',
        '--eps-clip', '0.2', '--eps-clip-high', '0.2', '--entropy-coef', '0', '--kl-coef', '0', '--kl-loss-coef', '0',
        '--get-mismatch-metrics', '--custom-config-path', str(ROOT/'mismatch-metrics.yaml'),
        '--custom-tis-function-path', 'examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp',
        '--optimizer', 'adam', '--lr', '1e-6', '--lr-decay-style', 'constant',
        '--weight-decay', '0', '--adam-beta1', '0.9', '--adam-beta2', '0.999', '--clip-grad', '1', '--bf16',
        '--use-precision-aware-optimizer', '--tensor-model-parallel-size', '1', '--pipeline-model-parallel-size', '1',
        '--context-parallel-size', '1', '--expert-model-parallel-size', '8', '--expert-tensor-parallel-size', '1',
        '--recompute-granularity', 'full', '--recompute-method', 'uniform', '--recompute-num-layers', '1',
        '--use-dynamic-batch-size', '--max-tokens-per-gpu', '4096', '--balance-data',
        '--attention-dropout', '0', '--hidden-dropout', '0', '--accumulate-allreduce-grads-in-fp32',
        '--attention-softmax-in-fp32', '--attention-backend', 'flash',
        '--actor-num-nodes', '1', '--actor-num-gpus-per-node', '8', '--num-gpus-per-node', '8',
        '--rollout-num-gpus', '6', '--rollout-num-gpus-per-engine', '2', '--sglang-ep-size', '2',
        '--sglang-moe-runner-backend', 'triton', '--sglang-mem-fraction-static', '0.65',
        '--sglang-max-running-requests', '256', '--sglang-server-concurrency', '128',
        '--sglang-context-length', '2048', '--sglang-mamba-scheduler-strategy', 'extra_buffer',
        '--sglang-enable-metrics', '--no-offload-train', '--no-offload-rollout',
        '--eval-interval', '10', '--eval-config', str(result/'eval.json'), '--eval-temperature', '0',
        '--n-samples-per-eval-prompt', '1', '--eval-max-response-len', '256',
        '--update-weights-interval', '1', '--use-tensorboard', '--tb-project-name', str(result/'tensorboard'),
        '--tb-experiment-name', 'slime-mopd', '--save-debug-rollout-data', str(result/'native-rollouts'/'rollout_{rollout_id}.pt'),
    ]
    (result/'argv.json').write_text(json.dumps(args, indent=2))
    return args


if __name__ == '__main__':
    import sys
    result = Path(sys.argv[1])
    result.mkdir(parents=True, exist_ok=True)
    build(result)
