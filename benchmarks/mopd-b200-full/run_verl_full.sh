#!/usr/bin/env bash
set -euo pipefail

bench_root=${BENCH_ROOT:-/dev/shm/opd-bench}
artifact_root=${ARTIFACT_ROOT:-$bench_root/artifacts/mopd-b200-full}
source_root="$bench_root/src/verl"
run_id=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
steps=${STEPS:-15}
run_dir="$bench_root/runs-full/verl-$run_id"
mkdir -p "$run_dir"

math_pid=
code_pid=
sampler_pid=
cleanup() {
  if [[ -n "$sampler_pid" ]]; then kill "$sampler_pid" 2>/dev/null || true; fi
  docker exec opd-verl-full bash -lc 'ray stop --force >/dev/null 2>&1 || true' 2>/dev/null || true
  docker restart opd-miles-full >/dev/null 2>&1 || true
  if [[ -n "$math_pid" ]]; then kill "$math_pid" 2>/dev/null || true; fi
  if [[ -n "$code_pid" ]]; then kill "$code_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local port=$1 pid=$2 log=$3
  for _ in $(seq 1 600); do
    curl -fsS "http://127.0.0.1:$port/health_generate" >/dev/null 2>&1 && return 0
    if ! kill -0 "$pid" 2>/dev/null; then tail -120 "$log"; return 1; fi
    sleep 1
  done
  tail -120 "$log"
  return 1
}

cd "$source_root"
patch_file="$artifact_root/verl_external_teachers.patch"
if git apply --reverse --check "$patch_file" >/dev/null 2>&1; then
  :
else
  git apply --check "$patch_file"
  git apply "$patch_file"
fi
padding_patch="$artifact_root/verl_padding_teacher_fields.patch"
if git apply --reverse --check "$padding_patch" >/dev/null 2>&1; then
  :
else
  git apply --check "$padding_patch"
  git apply "$padding_patch"
fi
docker restart opd-miles-full >/dev/null

process_start_epoch=$(date +%s)
"$artifact_root/gpu_sampler.sh" "$run_dir/gpu.csv" &
sampler_pid=$!
docker exec -i opd-miles-full bash -lc "exec env CUDA_VISIBLE_DEVICES=6 python3 -m sglang.launch_server \
  --model-path '$bench_root/models-full/math-teacher' --host 0.0.0.0 --port 13141 \
  --tp 1 --chunked-prefill-size 4096 --mem-fraction-static 0.70" \
  > "$run_dir/math-teacher.log" 2>&1 &
math_pid=$!
docker exec -i opd-miles-full bash -lc "exec env CUDA_VISIBLE_DEVICES=7 python3 -m sglang.launch_server \
  --model-path '$bench_root/models-full/code-teacher' --host 0.0.0.0 --port 13142 \
  --tp 1 --chunked-prefill-size 4096 --mem-fraction-static 0.70" \
  > "$run_dir/code-teacher.log" 2>&1 &
code_pid=$!
wait_for_server 13141 "$math_pid" "$run_dir/math-teacher.log"
wait_for_server 13142 "$code_pid" "$run_dir/code-teacher.log"
teachers_ready_epoch=$(date +%s)

teacher_urls='{"math":"http://127.0.0.1:13141/generate","code":"http://127.0.0.1:13142/generate"}'
training_start_epoch=$(date +%s)
docker exec -i \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  -e HF_HOME="$bench_root/cache/huggingface" \
  -e WANDB_MODE=disabled -e PYTHONUNBUFFERED=1 \
  -e VERL_EXTERNAL_TEACHER_URLS="$teacher_urls" \
  opd-verl-full bash -lc "
    set -euo pipefail
    cd /workspace/verl
    ray stop --force >/dev/null 2>&1 || true
    python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
      data.train_files='$bench_root/data-full/verl_train.parquet' \
      data.val_files='$bench_root/data-full/verl_eval.parquet' \
      data.train_batch_size=16 data.max_prompt_length=1024 data.max_response_length=16384 \
      data.filter_overlong_prompts=True data.truncation=error data.shuffle=False data.seed=42 \
      actor_rollout_ref.model.path='$bench_root/models-full/student' \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.use_torch_compile=False \
      actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.weight_decay=0.1 \
      actor_rollout_ref.actor.optim.betas='[0.9,0.98]' actor_rollout_ref.actor.optim.clip_grad=1.0 \
      actor_rollout_ref.actor.ppo_mini_batch_size=6 actor_rollout_ref.actor.use_dynamic_bsz=True \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.70 actor_rollout_ref.rollout.n=4 \
      actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
      actor_rollout_ref.rollout.max_model_len=17408 \
      actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
      reward.custom_reward_function.path='$artifact_root/zero_reward.py' \
      reward.custom_reward_function.name=compute_score \
      trainer.logger='[\"console\"]' trainer.project_name=mopd-b200-full \
      trainer.experiment_name='verl-$run_id' trainer.n_gpus_per_node=6 trainer.nnodes=1 \
      trainer.val_before_train=False trainer.save_freq=-1 trainer.test_freq=-1 \
      trainer.total_training_steps='$steps' trainer.resume_mode=disable \
      trainer.default_local_dir='$run_dir/checkpoints' \
      distillation.enabled=True distillation.n_gpus_per_node=2 distillation.nnodes=1 \
      distillation.teacher_key=data_source \
      +distillation.teacher_models.math_teacher.key=math \
      +distillation.teacher_models.math_teacher.model_path='$bench_root/models-full/math-teacher' \
      +distillation.teacher_models.math_teacher.num_replicas=1 \
      +distillation.teacher_models.math_teacher.inference.name=sglang \
      +distillation.teacher_models.math_teacher.inference.tensor_model_parallel_size=1 \
      +distillation.teacher_models.math_teacher.inference.max_model_len=17408 \
      +distillation.teacher_models.code_teacher.key=code \
      +distillation.teacher_models.code_teacher.model_path='$bench_root/models-full/code-teacher' \
      +distillation.teacher_models.code_teacher.num_replicas=1 \
      +distillation.teacher_models.code_teacher.inference.name=sglang \
      +distillation.teacher_models.code_teacher.inference.tensor_model_parallel_size=1 \
      +distillation.teacher_models.code_teacher.inference.max_model_len=17408 \
      distillation.distillation_loss.loss_mode=k1 \
      distillation.distillation_loss.use_task_rewards=False \
      distillation.distillation_loss.use_policy_gradient=True \
      distillation.distillation_loss.policy_loss_mode=vanilla \
      distillation.distillation_loss.clip_ratio_low=0.2 \
      distillation.distillation_loss.clip_ratio_high=0.28 \
      distillation.distillation_loss.loss_max_clamp=10.0 \
      distillation.distillation_loss.log_prob_min_clamp=-10.0 \
      hydra.run.dir='$run_dir/hydra'
  " > "$run_dir/run.log" 2>&1
end_epoch=$(date +%s)

printf '%s\n' \
  "framework=verl" "run_id=$run_id" "start_epoch=$process_start_epoch" \
  "teachers_ready_epoch=$teachers_ready_epoch" "training_start_epoch=$training_start_epoch" \
  "end_epoch=$end_epoch" "teacher_start_seconds=$((teachers_ready_epoch-process_start_epoch))" \
  "framework_wall_seconds=$((end_epoch-training_start_epoch))" \
  "wall_seconds=$((end_epoch-process_start_epoch))" > "$run_dir/wall_time.env"
printf '%s\n' "$run_dir"
