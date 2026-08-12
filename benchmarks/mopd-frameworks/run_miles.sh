#!/usr/bin/env bash
set -euo pipefail

bench_root=${BENCH_ROOT:-/root/opd-bench}
artifact_root="$bench_root/artifacts/mopd-frameworks"
run_id=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
steps=${STEPS:-15}
teacher_mode=${TEACHER_MODE:-multi}
run_dir="$bench_root/mopd/runs/miles-$run_id"
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
  if [[ -n "$math_pid" ]]; then kill "$math_pid" 2>/dev/null || true; fi
  if [[ -n "$code_pid" ]]; then kill "$code_pid" 2>/dev/null || true; fi
  CUDA_VISIBLE_DEVICES=0 ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cd /workspace/miles
export HF_HOME="$bench_root/cache/huggingface"
export PYTHONUNBUFFERED=1

process_start_epoch=$(date +%s)
"$artifact_root/gpu_sampler.sh" "$run_dir/gpu.csv" &
sampler_pid=$!

CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
  --model-path "$bench_root/mopd/models/math-teacher" \
  --host 0.0.0.0 --port 13141 --tp 1 \
  --chunked-prefill-size 4096 --mem-fraction-static 0.38 \
  --disable-cuda-graph > "$run_dir/math-teacher.log" 2>&1 &
math_pid=$!
for _ in $(seq 1 300); do
  curl -fsS http://127.0.0.1:13141/health_generate >/dev/null 2>&1 && break
  if ! kill -0 "$math_pid" 2>/dev/null; then tail -100 "$run_dir/math-teacher.log"; exit 1; fi
  sleep 1
done
curl -fsS http://127.0.0.1:13141/health_generate >/dev/null

if [[ "$teacher_mode" == multi ]]; then
  CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
    --model-path "$bench_root/mopd/models/code-teacher" \
    --host 0.0.0.0 --port 13142 --tp 1 \
    --chunked-prefill-size 4096 --mem-fraction-static 0.38 \
    --disable-cuda-graph > "$run_dir/code-teacher.log" 2>&1 &
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
  teacher_args=(--opd-teacher-urls math=http://127.0.0.1:13141/generate code=http://127.0.0.1:13141/generate)
else
  teacher_args=(--opd-teacher-urls math=http://127.0.0.1:13141/generate code=http://127.0.0.1:13142/generate)
fi

export CUDA_VISIBLE_DEVICES=0
ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port=8265 > "$run_dir/ray.log" 2>&1
ray_ready_epoch=$(date +%s)

source scripts/models/qwen2.5-0.5B.sh
training_start_epoch=$(date +%s)
ray job submit --address=http://127.0.0.1:8265 \
  --runtime-env-json="{\"env_vars\":{\"PYTHONPATH\":\"/root/Megatron-LM/\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\"}}" \
  -- python3 train.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node 1 --colocate \
  --hf-checkpoint "$bench_root/mopd/models/student" \
  --ref-load "$bench_root/mopd/models/student_torch_dist" \
  --prompt-data "$bench_root/mopd/data/miles_train.jsonl" \
  --input-key messages --label-key label --apply-chat-template --rollout-shuffle \
  --num-rollout "$steps" --rollout-batch-size 16 --n-samples-per-prompt 4 \
  --rollout-max-response-len 256 --rollout-temperature 1.0 --rollout-top-p 1.0 \
  --seed 42 --global-batch-size 64 --balance-data \
  --custom-rm-path miles.rollout.on_policy_distillation.reward_func \
  --custom-reward-post-process-path miles.rollout.on_policy_distillation.post_process_rewards \
  "${teacher_args[@]}" \
  --opd-teacher-key opd_teacher \
  --advantage-estimator grpo --use-opd --opd-type sglang \
  --opd-kl-coef 1.0 --opd-log-prob-top-k 0 \
  --use-kl-loss --kl-loss-coef 0.0 --kl-loss-type low_var_kl \
  --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --clip-grad 1.0 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 --sequence-parallel \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 8192 \
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.5 \
  --sglang-enable-metrics --attention-dropout 0.0 --hidden-dropout 0.0 \
  --bf16 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --attention-backend flash "${MODEL_ARGS[@]}" \
  2>&1 | tee "$run_dir/run.log"
end_epoch=$(date +%s)

printf '%s\n' \
  "framework=miles" "teacher_mode=$teacher_mode" "run_id=$run_id" "start_epoch=$process_start_epoch" \
  "teachers_ready_epoch=$teachers_ready_epoch" "ray_ready_epoch=$ray_ready_epoch" \
  "training_start_epoch=$training_start_epoch" "end_epoch=$end_epoch" \
  "teacher_start_seconds=$((teachers_ready_epoch - process_start_epoch))" \
  "framework_wall_seconds=$((end_epoch - training_start_epoch))" \
  "wall_seconds=$((end_epoch - process_start_epoch))" > "$run_dir/wall_time.env"

printf '%s\n' "$run_dir"
