"""verl recipe: multi-teacher OPD on verl's fully async trainer, built from `shared.recipe`.

Follows upstream `examples/on_policy_distillation_trainer` (teachers routed by `data_source`,
sampled-token k1 reverse KL applied as a policy gradient, no task reward), run on
`verl.experimental.fully_async_policy` with FSDP training and vLLM serving. The learner role runs
this from the campaign's `source/` and saves the overrides as `argv.json`. `compute_score` is verl's
reward hook; its score is diagnostic only.
"""
import json
import os
import subprocess
import sys

from shared import recipe
from shared.slurm import DATA, ROOT, result_dir, site

MAX_IN_FLIGHT = 256  # Concurrent policy requests, as in the other frameworks.
TEACHER_GPUS = len(recipe.TEACHER_GPU)


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
    policy_engines = len(r.POLICY_GPUS) // r.POLICY_GPUS_PER_ENGINE
    # As in upstream's example, one teacher uses the default `teacher_model` entry; several are added
    # as named entries routed by `data_source`. The +1 in max_model_len fits the student's prompt and
    # full response plus one generated teacher token.
    inference = ['inference.name=vllm', 'inference.tensor_model_parallel_size=1',
                 'inference.gpu_memory_utilization=0.8', f'inference.max_model_len={r.CONTEXT_LENGTH + 1}']
    if len(r.DOMAINS) == 1:
        entry = 'distillation.teacher_models.teacher_model'
        teachers = [f'{entry}.model_path={s["teachers"]}/{r.DOMAINS[0]}', *(f'{entry}.{x}' for x in inference)]
    else:
        teachers = [value for d in r.DOMAINS for value in (
            f'+distillation.teacher_models.{d}.key={d}',
            f'+distillation.teacher_models.{d}.model_path={s["teachers"]}/{d}',
            f'+distillation.teacher_models.{d}.num_replicas=1',
            *(f'+distillation.teacher_models.{d}.{x}' for x in inference))]
    return [
        # Data: one prompt at a time; with two domains, the mixed file interleaves them.
        f'data.train_files=["{train}"]', f'data.val_files={json.dumps([str(p) for p in dev])}',
        'data.prompt_key=prompt', 'data.return_raw_chat=True', 'data.train_batch_size=0', 'data.gen_batch_size=1',
        f'data.max_prompt_length={prompt_tokens}', f'data.max_response_length={r.MAX_RESPONSE_TOKENS}',
        'data.filter_overlong_prompts=True', 'data.truncation=error', 'data.shuffle=True',
        f'+data.apply_chat_template_kwargs.enable_thinking={r.ENABLE_THINKING}',
        # Objective: sampled-token reverse KL to the routed teacher as the only reward.
        'algorithm.adv_estimator=grpo', 'algorithm.use_kl_in_reward=False',
        'distillation.enabled=True', f'distillation.n_gpus_per_node={TEACHER_GPUS}', 'distillation.nnodes=1',
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
        f'actor_rollout_ref.actor.ppo_max_token_len_per_gpu={r.MAX_TOKENS_PER_GPU}',
        # Policy engines: vLLM on their own GPUs, weights synced after every update.
        'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.mode=async',
        f'actor_rollout_ref.rollout.n={r.SAMPLES_PER_PROMPT}',
        f'actor_rollout_ref.rollout.tensor_model_parallel_size={r.POLICY_GPUS_PER_ENGINE}',
        'actor_rollout_ref.rollout.gpu_memory_utilization=0.8',
        f'actor_rollout_ref.rollout.max_model_len={r.CONTEXT_LENGTH + 1}',
        'actor_rollout_ref.rollout.temperature=1.0', 'actor_rollout_ref.rollout.top_p=1.0',
        f'actor_rollout_ref.rollout.prompt_length={prompt_tokens}',
        f'actor_rollout_ref.rollout.response_length={r.MAX_RESPONSE_TOKENS}',
        'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True',
        f'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={r.MAX_TOKENS_PER_GPU}',
        'actor_rollout_ref.rollout.val_kwargs.do_sample=False', 'actor_rollout_ref.rollout.val_kwargs.n=1',
        'rollout.nnodes=1', f'rollout.n_gpus_per_node={len(r.POLICY_GPUS)}', f'rollout.n={r.SAMPLES_PER_PROMPT}',
        f'rollout.total_rollout_steps={r.UPDATES * r.PROMPTS_PER_UPDATE}',
        # A weight sync after every update and at most MAX_POLICY_LAG batches of policy lag.
        f'async_training.staleness_threshold={r.MAX_POLICY_LAG}', 'async_training.trigger_parameter_sync_step=1',
        # Each weight sync aborts in-flight requests; partial rollout (the trainer's default) resumes
        # them instead of returning them truncated.
        'async_training.require_batches=1', 'async_training.partial_rollout=True',
        f'async_training.concurrent_samples_per_replica={-(-MAX_IN_FLIGHT // policy_engines)}',
        # Trainer, evaluation and checkpoints.
        'trainer.nnodes=1', f'trainer.n_gpus_per_node={r.TRAINER_GPUS}', 'trainer.total_epochs=1', 'trainer.balance_batch=True',
        'trainer.val_before_train=True', f'trainer.test_freq={r.EVAL_INTERVAL}', f'trainer.save_freq={r.SAVE_INTERVAL}',
        f'trainer.default_local_dir={ROOT / "checkpoints" / result.name}', 'trainer.logger=["console","tensorboard"]',
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
