"""verl recipe: multi-teacher OPD on verl's fully async trainer, built from `shared.recipe`.

Follows upstream `examples/on_policy_distillation_trainer` (teachers routed by `data_source`,
sampled-token k1 reverse KL applied as a policy gradient, no task reward), run on
`verl.experimental.fully_async_policy` with FSDP training and vLLM serving. The learner role runs
this from the campaign's `source/` and saves the overrides as `argv.json`. `compute_score` is verl's
reward hook; its score is diagnostic only. Each trained batch, with its scores, is written to
`<result>/rollouts/<step>.jsonl`.
"""
import json
import os
import subprocess
import sys

from shared import recipe
from shared.slurm import DATA, ROOT, result_dir, site

# Concurrent policy requests. b200x2: 256, as in the other frameworks. h100x3: 160 (1.25 batches), as in
# Miles run 535; at policy lag 1 a deeper pipeline mostly produces samples that are dropped as stale.
MAX_IN_FLIGHT = 160 if recipe.H100 else 256
# verl's staleness_threshold is a fraction of a batch: the rollouter generates at most (1 + threshold)
# batches per weight sync. 1 matches Miles' policy lag 1 with a one-batch buffer; concurrency is capped
# separately by MAX_IN_FLIGHT. (0.25 in job 544 also capped generation at 160 samples per sync, so the
# rollouter stopped every update and the trainer waited a median 78 s for rollouts.)
STALENESS_THRESHOLD = recipe.MAX_POLICY_LAG
# Tokens per GPU per micro-batch. h100x3 offloads the parameters to the CPU (below), so every micro-batch
# copies them back to the GPU; job 544 peaked at 14 GB of 80 GB with MAX_TOKENS_PER_GPU split over the two
# sequence-parallel GPUs, so each GPU takes a full MAX_TOKENS_PER_GPU instead (half the micro-batches).
# This changes only the batching, not the gradient.
MICRO_BATCH_TOKENS_PER_GPU = recipe.MAX_TOKENS_PER_GPU if recipe.H100 else recipe.MAX_TOKENS_PER_GPU // recipe.TRAINER_CONTEXT_PARALLEL
# Teacher pool: b200x2 puts every teacher on the generation node, one GPU each; h100x3 puts one TP2
# teacher on each generation node. verl splits the pool per teacher, in DOMAINS order.
TEACHER_NODES = len({node for node, _ in recipe.TEACHER_PLACEMENT.values()})
TEACHER_GPUS_PER_NODE = len(recipe.DOMAINS) * recipe.TEACHER_GPUS_PER_ENGINE // TEACHER_NODES


def write_parquet(result):
    """Write parquet rows for verl: chat prompt, teacher routing key and scoring label."""
    import pandas

    def rows(name):
        for line in (DATA / name).read_text().splitlines():
            row = json.loads(line)
            yield {'prompt': row['prompt'], 'data_source': row['metadata']['domain'],
                   'reward_model': {'style': 'rule', 'ground_truth': row['label']},
                   'extra_info': {'index': row['metadata']['source_index']}}

    out = result / 'verl-data'
    out.mkdir(exist_ok=True)
    pandas.DataFrame(rows(recipe.TRAIN_FILE)).to_parquet(out / 'train.parquet')
    dev = [out / f'{d}-dev.parquet' for d in recipe.DOMAINS]
    for domain, path in zip(recipe.DOMAINS, dev):
        pandas.DataFrame(rows(recipe.data_file(domain, 'dev'))).to_parquet(path)
    return out / 'train.parquet', dev


def overrides(result, train, dev):
    s, r = site(), recipe
    prompt_tokens = r.CONTEXT_LENGTH - r.MAX_RESPONSE_TOKENS
    policy_engines = r.ROLLOUT_GPUS // r.POLICY_GPUS_PER_ENGINE  # across every generation node
    # As in upstream's example, one teacher uses the default `teacher_model` entry; several are added
    # as named entries routed by `data_source`. The +1 in max_model_len fits the student's prompt and
    # full response plus one generated teacher token.
    # disable_log_stats=False turns on vLLM's Prometheus request metrics (latency, tokens), which
    # shared/capture.py scrapes; verl turns them off by default.
    inference = ['inference.name=vllm', f'inference.tensor_model_parallel_size={r.TEACHER_GPUS_PER_ENGINE}',
                 'inference.gpu_memory_utilization=0.8', f'inference.max_model_len={r.CONTEXT_LENGTH + 1}',
                 'inference.disable_log_stats=False']
    if len(r.DOMAINS) == 1:
        entry = 'distillation.teacher_models.teacher_model'
        teachers = [f'{entry}.model_path={s["teachers"]}/{r.DOMAINS[0]}', *(f'{entry}.{x}' for x in inference)]
    else:
        # Named entries do not inherit the `teacher_model` block of distillation.yaml; they start from
        # RolloutConfig, whose load_format is "dummy" (random weights) and whose prompt and response
        # lengths are 512. Set the values the default entry would have.
        named = [*inference, 'inference.load_format=auto', 'inference.enforce_eager=True',
                 f'inference.prompt_length={prompt_tokens}', f'inference.response_length={r.MAX_RESPONSE_TOKENS}']
        teachers = [value for d in r.DOMAINS for value in (
            f'+distillation.teacher_models.{d}.key={d}',
            f'+distillation.teacher_models.{d}.model_path={s["teachers"]}/{d}',
            f'+distillation.teacher_models.{d}.num_replicas=1',
            *(f'+distillation.teacher_models.{d}.{x}' for x in named))]
    return [
        # Data: one prompt at a time; with two domains, the mixed file interleaves them.
        f'data.train_files=["{train}"]', f'data.val_files={json.dumps([str(p) for p in dev])}',
        'data.prompt_key=prompt', 'data.return_raw_chat=True', 'data.train_batch_size=0', 'data.gen_batch_size=1',
        f'data.max_prompt_length={prompt_tokens}', f'data.max_response_length={r.MAX_RESPONSE_TOKENS}',
        'data.filter_overlong_prompts=True', 'data.truncation=error', 'data.shuffle=True',
        f'+data.apply_chat_template_kwargs.enable_thinking={r.ENABLE_THINKING}',
        # Objective: sampled-token reverse KL to the routed teacher as the only reward.
        'algorithm.adv_estimator=grpo', 'algorithm.use_kl_in_reward=False',
        'distillation.enabled=True', f'distillation.n_gpus_per_node={TEACHER_GPUS_PER_NODE}',
        f'distillation.nnodes={TEACHER_NODES}',
        'distillation.teacher_key=data_source', *teachers,
        'distillation.distillation_loss.loss_mode=k1', 'distillation.distillation_loss.use_policy_gradient=True',
        'distillation.distillation_loss.use_task_rewards=False',
        'distillation.distillation_loss.loss_max_clamp=10.0', 'distillation.distillation_loss.log_prob_min_clamp=-10.0',
        f'reward.custom_reward_function.path={ROOT / "train.py"}', 'reward.custom_reward_function.name=compute_score',
        # Student model and FSDP trainer, as in upstream run_qwen3_8b_mopd_fsdp.sh; no sequence parallelism.
        f'actor_rollout_ref.model.path={s["base_model"]}', 'actor_rollout_ref.model.trust_remote_code=True',
        'actor_rollout_ref.model.use_remove_padding=True', 'actor_rollout_ref.model.enable_gradient_checkpointing=True',
        'actor_rollout_ref.actor.use_torch_compile=True', 'actor_rollout_ref.hybrid_engine=False',
        f'actor_rollout_ref.actor.optim.lr={r.LEARNING_RATE}', 'actor_rollout_ref.actor.optim.weight_decay=0',
        'actor_rollout_ref.actor.optim.clip_grad=1.0', 'actor_rollout_ref.actor.entropy_coeff=0',
        'actor_rollout_ref.actor.use_kl_loss=False',
        f'actor_rollout_ref.actor.ppo_mini_batch_size={r.PROMPTS_PER_UPDATE}',
        'actor_rollout_ref.actor.use_dynamic_bsz=True',
        # verl packs this times the sequence-parallel size into each micro-batch; see MICRO_BATCH_TOKENS_PER_GPU.
        f'actor_rollout_ref.actor.ppo_max_token_len_per_gpu={MICRO_BATCH_TOKENS_PER_GPU}',
        # h100x3: two GPUs share each sequence (Ulysses), the counterpart of Miles' context parallelism 2.
        f'actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size={r.TRAINER_CONTEXT_PARALLEL}',
        # h100x3: the counterpart of Miles' CPU Adam is FSDP2's CPUOffloadPolicy: fp32 parameters, gradients
        # and Adam states stay in pinned CPU memory and the optimizer step runs on the CPU. verl's
        # `optimizer_offload` is not enough: it moves the Adam states back to the GPU for the step, which ran
        # out of memory in update 1 (job 543). Entropy checkpointing as in upstream's Qwen3.5 OPD examples.
        *([f'actor_rollout_ref.actor.strategy=fsdp2', 'actor_rollout_ref.actor.fsdp_config.offload_policy=True',
           'actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True']
          if r.TRAINER_OPTIMIZER_CPU_OFFLOAD else []),
        # Policy engines: vLLM on their own GPUs, weights synced after every update.
        'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.mode=async',
        f'actor_rollout_ref.rollout.n={r.SAMPLES_PER_PROMPT}',
        f'actor_rollout_ref.rollout.tensor_model_parallel_size={r.POLICY_GPUS_PER_ENGINE}',
        'actor_rollout_ref.rollout.gpu_memory_utilization=0.8', 'actor_rollout_ref.rollout.disable_log_stats=False',
        f'actor_rollout_ref.rollout.max_model_len={r.CONTEXT_LENGTH + 1}',
        'actor_rollout_ref.rollout.temperature=1.0', 'actor_rollout_ref.rollout.top_p=1.0',
        f'actor_rollout_ref.rollout.prompt_length={prompt_tokens}',
        f'actor_rollout_ref.rollout.response_length={r.MAX_RESPONSE_TOKENS}',
        'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True',
        f'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={MICRO_BATCH_TOKENS_PER_GPU}',
        'actor_rollout_ref.rollout.val_kwargs.do_sample=False', 'actor_rollout_ref.rollout.val_kwargs.n=1',
        f'rollout.nnodes={r.GENERATION_NODES}', f'rollout.n_gpus_per_node={len(r.POLICY_GPUS)}',
        f'rollout.n={r.SAMPLES_PER_PROMPT}',
        f'rollout.total_rollout_steps={r.UPDATES * r.PROMPTS_PER_UPDATE}',
        # A weight sync after every update; see STALENESS_THRESHOLD.
        f'async_training.staleness_threshold={STALENESS_THRESHOLD}', 'async_training.trigger_parameter_sync_step=1',
        # Each weight sync aborts in-flight requests; partial rollout (the trainer's default) resumes
        # them instead of returning them truncated.
        'async_training.require_batches=1', 'async_training.partial_rollout=True',
        f'async_training.concurrent_samples_per_replica={-(-MAX_IN_FLIGHT // policy_engines)}',
        # Trainer, evaluation and checkpoints.
        'trainer.nnodes=1', f'trainer.n_gpus_per_node={r.TRAINER_GPUS}', 'trainer.total_epochs=1', 'trainer.balance_batch=True',
        # h100x3 evaluates only after the last update, as Miles run 535 does; greedy 30k-token evaluation
        # pauses the rollouter, and tools/benchmark.py evaluates every saved checkpoint offline.
        f'trainer.val_before_train={not r.H100}', f'trainer.test_freq={r.UPDATES if r.H100 else r.EVAL_INTERVAL}',
        f'trainer.save_freq={r.SAVE_INTERVAL}',
        f'trainer.default_local_dir={ROOT / "checkpoints" / result.name}', 'trainer.logger=["console","tensorboard"]',
        f'trainer.rollout_data_dir={result / "rollouts"}',
        'trainer.project_name=opd', f'trainer.experiment_name={result.name}']


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    from shared.scoring import score
    return score(solution_str, ground_truth)


if __name__ == '__main__':
    result = result_dir()
    args = overrides(result, *write_parquet(result))
    (result / 'argv.json').write_text(json.dumps(args, indent=2))
    command = [sys.executable, '-m', 'verl.experimental.fully_async_policy.fully_async_main',
               '--config-name=fully_async_ppo_trainer', *args]
    # verl's config search path is relative to the source root.
    sys.exit(subprocess.run(command, cwd=ROOT / 'source', env={**os.environ, 'TENSORBOARD_DIR': str(result / 'tensorboard')}).returncode)
