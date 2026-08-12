#!/usr/bin/env bash
set -euo pipefail

bench_root=${BENCH_ROOT:-/root/opd-bench}
run_id=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
steps=${STEPS:-15}
run_dir="$bench_root/runs/verl-$run_id"
mkdir -p "$run_dir"

sampler_pid=
cleanup() {
  if [[ -n "$sampler_pid" ]]; then kill "$sampler_pid" 2>/dev/null || true; fi
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cd /workspace/verl
export HF_HOME="$bench_root/cache/huggingface"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

ray stop --force >/dev/null 2>&1 || true
process_start_epoch=$(date +%s)
"$bench_root/artifacts/opd-frameworks/gpu_sampler.sh" "$run_dir/gpu.csv" &
sampler_pid=$!

training_start_epoch=$(date +%s)
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$bench_root/data/reverse-text/train.parquet" \
  data.val_files="$bench_root/data/reverse-text/eval.parquet" \
  data.train_batch_size=8 \
  data.max_prompt_length=512 \
  data.max_response_length=128 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=False \
  data.seed=42 \
  actor_rollout_ref.model.path="$bench_root/models/student" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.optim.betas='[0.9,0.999]' \
  actor_rollout_ref.actor.optim.clip_grad=1.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.max_model_len=641 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  reward.custom_reward_function.path="$bench_root/artifacts/opd-frameworks/reverse_text_reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.logger='["console"]' \
  trainer.project_name=opd-framework-benchmark \
  trainer.experiment_name="verl-$run_id" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_training_steps="$steps" \
  trainer.resume_mode=disable \
  trainer.default_local_dir="$run_dir/checkpoints" \
  distillation.enabled=True \
  distillation.n_gpus_per_node=1 \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="$bench_root/models/teacher" \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.4 \
  distillation.teacher_models.teacher_model.inference.max_model_len=641 \
  distillation.distillation_loss.loss_mode=k1 \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.use_policy_gradient=True \
  distillation.distillation_loss.policy_loss_mode=vanilla \
  distillation.distillation_loss.clip_ratio_low=0.2 \
  distillation.distillation_loss.clip_ratio_high=0.28 \
  distillation.distillation_loss.loss_max_clamp=10.0 \
  distillation.distillation_loss.log_prob_min_clamp=-10.0 \
  hydra.run.dir="$run_dir/hydra" \
  > "$run_dir/run.log" 2>&1
end_epoch=$(date +%s)

printf '%s\n' \
  "framework=verl" \
  "run_id=$run_id" \
  "start_epoch=$process_start_epoch" \
  "training_start_epoch=$training_start_epoch" \
  "end_epoch=$end_epoch" \
  "framework_wall_seconds=$((end_epoch - training_start_epoch))" \
  "wall_seconds=$((end_epoch - process_start_epoch))" \
  > "$run_dir/wall_time.env"

printf '%s\n' "$run_dir"
