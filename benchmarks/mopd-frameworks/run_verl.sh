#!/usr/bin/env bash
set -euo pipefail

# Host-side launcher: verl trains in opd-verl while the two SGLang teachers run
# in opd-miles. Both containers use host networking and mount BENCH_ROOT.
bench_root=${BENCH_ROOT:-/root/opd-bench}
artifact_root="$bench_root/artifacts/mopd-frameworks"
source_root="$bench_root/src/verl"
run_id=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
steps=${STEPS:-15}
teacher_mode=${TEACHER_MODE:-multi}
run_dir="$bench_root/mopd/runs/verl-$run_id"
mkdir -p "$run_dir"

case "$teacher_mode" in
  single|multi) ;;
  *) printf 'TEACHER_MODE must be single or multi, got %s\n' "$teacher_mode" >&2; exit 2 ;;
esac

math_pid=
code_pid=
sampler_pid=
cleanup() {
  if [[ -n "$sampler_pid" ]]; then kill "$sampler_pid" 2>/dev/null || true; fi
  docker exec opd-verl bash -lc 'ray stop --force >/dev/null 2>&1 || true' || true
  # Restarting the dedicated serving container reaps SGLang's scheduler
  # subprocesses as well as the Python launchers, avoiding leaked GPU contexts.
  docker restart opd-miles >/dev/null 2>&1 || true
  if [[ -n "$math_pid" ]]; then kill "$math_pid" 2>/dev/null || true; fi
  if [[ -n "$code_pid" ]]; then kill "$code_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

cd "$source_root"
if git apply --reverse --check "$artifact_root/verl_external_teachers.patch" >/dev/null 2>&1; then
  : # Patch is already installed.
else
  git apply --check "$artifact_root/verl_external_teachers.patch"
  git apply "$artifact_root/verl_external_teachers.patch"
fi

# Start from an empty serving container so no stale endpoint can satisfy the
# health probe while a new server fails to bind.
docker restart opd-miles >/dev/null

process_start_epoch=$(date +%s)
"$artifact_root/gpu_sampler.sh" "$run_dir/gpu.csv" &
sampler_pid=$!

docker exec -i opd-miles bash -lc "exec env CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
  --model-path '$bench_root/mopd/models/math-teacher' --host 0.0.0.0 --port 13141 --tp 1 \
  --chunked-prefill-size 4096 --mem-fraction-static 0.38 --disable-cuda-graph" \
  > "$run_dir/math-teacher.log" 2>&1 &
math_pid=$!
for _ in $(seq 1 300); do
  curl -fsS http://127.0.0.1:13141/health_generate >/dev/null 2>&1 && break
  if ! kill -0 "$math_pid" 2>/dev/null; then tail -100 "$run_dir/math-teacher.log"; exit 1; fi
  sleep 1
done
curl -fsS http://127.0.0.1:13141/health_generate >/dev/null

if [[ "$teacher_mode" == multi ]]; then
  docker exec -i opd-miles bash -lc "exec env CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
    --model-path '$bench_root/mopd/models/code-teacher' --host 0.0.0.0 --port 13142 --tp 1 \
    --chunked-prefill-size 4096 --mem-fraction-static 0.38 --disable-cuda-graph" \
    > "$run_dir/code-teacher.log" 2>&1 &
  code_pid=$!
  for _ in $(seq 1 300); do
    curl -fsS http://127.0.0.1:13142/health_generate >/dev/null 2>&1 && break
    if ! kill -0 "$code_pid" 2>/dev/null; then tail -100 "$run_dir/code-teacher.log"; exit 1; fi
    sleep 1
  done
  curl -fsS http://127.0.0.1:13142/health_generate >/dev/null
fi
teachers_ready_epoch=$(date +%s)

if [[ "$teacher_mode" == single ]]; then
  teacher_urls_json='{"math":"http://127.0.0.1:13141/generate","code":"http://127.0.0.1:13141/generate"}'
else
  teacher_urls_json='{"math":"http://127.0.0.1:13141/generate","code":"http://127.0.0.1:13142/generate"}'
fi

training_start_epoch=$(date +%s)
docker exec -i \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_HOME="$bench_root/cache/huggingface" \
  -e WANDB_MODE=disabled \
  -e PYTHONUNBUFFERED=1 \
  -e VERL_EXTERNAL_TEACHER_URLS="$teacher_urls_json" \
  opd-verl bash -lc "
    cd /workspace/verl
    ray stop --force >/dev/null 2>&1 || true
    python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
      data.train_files='$bench_root/mopd/data/verl_train.parquet' \
      data.val_files='$bench_root/mopd/data/verl_eval.parquet' \
      data.train_batch_size=16 data.max_prompt_length=1024 data.max_response_length=256 \
      data.filter_overlong_prompts=True data.truncation=error data.shuffle=False data.seed=42 \
      actor_rollout_ref.model.path='$bench_root/mopd/models/student' \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.use_torch_compile=False \
      actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.weight_decay=0.1 \
      actor_rollout_ref.actor.optim.betas='[0.9,0.98]' actor_rollout_ref.actor.optim.clip_grad=1.0 \
      actor_rollout_ref.actor.ppo_mini_batch_size=16 actor_rollout_ref.actor.use_dynamic_bsz=True \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.5 actor_rollout_ref.rollout.n=4 \
      actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
      actor_rollout_ref.rollout.max_model_len=1281 \
      actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
      reward.custom_reward_function.path='$artifact_root/zero_reward.py' \
      reward.custom_reward_function.name=compute_score \
      trainer.logger='[\"console\"]' trainer.project_name=mopd-framework-benchmark \
      trainer.experiment_name='verl-$run_id' trainer.n_gpus_per_node=1 trainer.nnodes=1 \
      trainer.val_before_train=False trainer.save_freq=-1 trainer.test_freq=-1 \
      trainer.total_training_steps='$steps' trainer.resume_mode=disable \
      trainer.default_local_dir='$run_dir/checkpoints' \
      distillation.enabled=True distillation.n_gpus_per_node=2 distillation.nnodes=1 \
      distillation.teacher_key=data_source \
      +distillation.teacher_models.math_teacher.key=math \
      +distillation.teacher_models.math_teacher.model_path='$bench_root/mopd/models/math-teacher' \
      +distillation.teacher_models.math_teacher.num_replicas=1 \
      +distillation.teacher_models.math_teacher.inference.name=sglang \
      +distillation.teacher_models.math_teacher.inference.tensor_model_parallel_size=1 \
      +distillation.teacher_models.math_teacher.inference.max_model_len=1281 \
      +distillation.teacher_models.code_teacher.key=code \
      +distillation.teacher_models.code_teacher.model_path='$bench_root/mopd/models/code-teacher' \
      +distillation.teacher_models.code_teacher.num_replicas=1 \
      +distillation.teacher_models.code_teacher.inference.name=sglang \
      +distillation.teacher_models.code_teacher.inference.tensor_model_parallel_size=1 \
      +distillation.teacher_models.code_teacher.inference.max_model_len=1281 \
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
  "framework=verl" "teacher_mode=$teacher_mode" "run_id=$run_id" "start_epoch=$process_start_epoch" \
  "teachers_ready_epoch=$teachers_ready_epoch" "training_start_epoch=$training_start_epoch" \
  "end_epoch=$end_epoch" "teacher_start_seconds=$((teachers_ready_epoch - process_start_epoch))" \
  "framework_wall_seconds=$((end_epoch - training_start_epoch))" \
  "wall_seconds=$((end_epoch - process_start_epoch))" > "$run_dir/wall_time.env"

printf '%s\n' "$run_dir"
