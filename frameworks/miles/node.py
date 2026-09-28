"""Miles campaign entry point: `node.py` runs the controller, `node.py ROLE` one container role.

The shared two-node runtime is `shared/container.py`; this adds the Miles
recipe: a fully async, sampled-token MOPD launch of the pinned upstream script.
"""
import json
import os
import shlex
import sys

from shared import recipe as r
from shared.container import ContainerNode, entry
from shared.slurm import DATA, ROOT, eval_config


class MilesNode(ContainerNode):
    placement_module = 'miles.ray.placement_group'
    # The upstream launcher joins the Ray cluster this role started instead of its own.
    env = {'MILES_SCRIPT_EXTERNAL_RAY': '1'}

    def placement(self, group):
        return [int(x) for x in group.pg_reordered_gpu_ids], group.pg

    def train_command(self, teacher_urls):
        (self.result / 'eval.json').write_text(json.dumps(eval_config({'mopd_evaluation': True}), indent=2))
        dev = [x for d in r.DOMAINS for x in (d, DATA / r.data_file(d, 'dev'))]
        # Appended after the pinned launcher's own arguments, so these take precedence. The first
        # group replaces the launcher's puzzle-specific tasks, thinking setting and 2k context.
        extra = ['--prompt-data', DATA / r.TRAIN_FILE, '--data-source-path', 'domain_balance.BalancedDataSource',
                 '--opd-domain-targets', *(f'{d}={1 / len(r.DOMAINS)}' for d in r.DOMAINS),
                 '--eval-prompt-data', *dev, '--eval-config', self.result / 'eval.json',
                 '--apply-chat-template-kwargs', json.dumps({'enable_thinking': r.ENABLE_THINKING}),
                 '--sglang-context-length', r.CONTEXT_LENGTH, '--context-parallel-size', '1',
                 '--log-probs-chunk-size', r.LOG_PROBS_CHUNK_SIZE,
                 # Match the Slime and Prime-RL optimizer; the launcher's own is wd 0.1, beta2 0.98.
                 '--weight-decay', '0', '--adam-beta2', '0.999',
                 # The MTP head is not trained (loss scale 0); dropping it frees memory at 32k.
                 '--mtp-num-layers', '0',
                 '--sglang-moe-runner-backend', 'triton', '--sglang-enable-metrics',
                 '--custom-async-data-buffer-path', 'domain_balance.BalancedAsyncBuffer',
                 '--custom-rm-path', 'opd_reward.reward_func',
                 '--async-max-concurrent-samples', '256', '--async-data-buffer-capacity-factor', '1',
                 '--max-weight-staleness', r.MAX_POLICY_LAG, '--update-weights-interval', '1',
                 '--save-debug-event-data', self.result / 'events',
                 # Hugging Face weights of each saved update, for benchmarking without a conversion.
                 '--save-hf', ROOT / 'checkpoints' / self.result.name / 'hf' / 'iter_{rollout_id}',
                 '--use-tensorboard', '--tb-project-name', self.result / 'tensorboard',
                 '--tb-experiment-name', 'async-mopd', '--no-offload-train', '--no-offload-rollout']
        worker_env = {'CAMPAIGN_RESULT': str(self.result), 'CAMPAIGN_ROOT': str(ROOT),
                      'CAMPAIGN_PYDEPS': os.environ['CAMPAIGN_PYDEPS'],
                      'TENSORBOARD_DIR': str(self.result / 'tensorboard'),
                      'PYTHONPATH': os.environ['PYTHONPATH'], 'NCCL_DEBUG': 'WARN'}
        # The pinned launcher's default supplies the recipe's learning rate.
        command = ['scripts/run_mopd_puzzles.py', '--mode', 'student', '--num-nodes', '2',
                   '--fully-async', '--no-colocate', '--use-rollout-logprobs', '--actor-gpus', r.TRAINER_GPUS,
                   '--megatron-path', self.site['megatron_source'], '--rollout-gpus', len(r.POLICY_GPUS),
                   '--rollout-gpus-per-engine', r.POLICY_GPUS_PER_ENGINE, '--model-dir', ROOT / 'models',
                   '--data-dir', DATA, '--checkpoint-dir', ROOT / 'checkpoints' / self.result.name,
                   '--teacher-urls', ' '.join(f'{domain}={url}' for domain, url in teacher_urls.items()),
                   # Sampled-token OPD: top-k candidate scoring is quadratic in response length.
                   '--candidate-top-k', '0', '--loss-mode', 'legacy', '--reward-refresh',
                   '--domain-balance', 'static', '--num-rollout', r.UPDATES, '--rollout-batch-size', r.PROMPTS_PER_UPDATE,
                   '--n-samples-per-prompt', r.SAMPLES_PER_PROMPT, '--global-batch-size', r.PROMPTS_PER_UPDATE,
                   '--max-response-len', r.MAX_RESPONSE_TOKENS, '--max-tokens-per-gpu', r.MAX_TOKENS_PER_GPU,
                   # A draft </answer> inside the thinking must not end the rollout.
                   '--no-stop-at-answer',
                   # Also bounds policy generations, which take minutes at 30k tokens.
                   '--teacher-timeout-seconds', '7200',
                   '--eval-interval', r.EVAL_INTERVAL, '--save-interval', r.SAVE_INTERVAL,
                   '--no-cleanup-processes', '--no-sparse-scoring', '--extra-env-vars', json.dumps(worker_env),
                   '--extra-args', shlex.join(map(str, extra))]
        return [sys.executable, *command], {}


if __name__ == '__main__':
    sys.exit(entry('miles', MilesNode))
