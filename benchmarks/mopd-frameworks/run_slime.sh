#!/usr/bin/env bash
set -euo pipefail

bench_root=${BENCH_ROOT:-/root/opd-bench}
artifact_root=${ARTIFACT_ROOT:-$bench_root/artifacts/mopd-frameworks}

# The same command works on the host and inside the dedicated setup_slime.sh container.
if [[ ${SLIME_IN_CONTAINER:-0} != 1 ]]; then
  if ! docker inspect opd-slime >/dev/null 2>&1; then
    printf 'Container opd-slime does not exist; run setup_slime.sh first.\n' >&2
    exit 1
  fi
  docker_args=(
    -i
    -e SLIME_IN_CONTAINER=1
    -e BENCH_ROOT="$bench_root"
    -e ARTIFACT_ROOT="$artifact_root"
  )
  for name in RUN_ID STEPS SAVE_INTERVAL TEACHER_MODE ACTOR_GPU MATH_TEACHER_GPU CODE_TEACHER_GPU \
    TEACHER_MEM_FRACTION ROLLOUT_MEM_FRACTION MATH_TEACHER_PORT CODE_TEACHER_PORT; do
    if [[ -v "$name" ]]; then
      docker_args+=(-e "$name=${!name}")
    fi
  done
  exec docker exec "${docker_args[@]}" opd-slime bash "$artifact_root/run_slime.sh"
fi

run_id=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
steps=${STEPS:-15}
save_interval=${SAVE_INTERVAL:-$steps}
teacher_mode=${TEACHER_MODE:-multi}
run_dir="$bench_root/mopd/runs/slime-$run_id"
model_root="$bench_root/mopd/models"
data_root="$bench_root/mopd/data"
math_port=${MATH_TEACHER_PORT:-13141}
code_port=${CODE_TEACHER_PORT:-13142}

if [[ ! "$steps" =~ ^[1-9][0-9]*$ ]]; then
  printf 'STEPS must be a positive integer, got %s\n' "$steps" >&2
  exit 2
fi
case "$teacher_mode" in
  single|multi) ;;
  *) printf 'TEACHER_MODE must be single or multi, got %s\n' "$teacher_mode" >&2; exit 2 ;;
esac
if [[ -e "$run_dir" ]]; then
  printf 'Run directory already exists: %s\n' "$run_dir" >&2
  exit 1
fi
mkdir -p "$run_dir/checkpoints"

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
if (( gpu_count < 1 )); then
  printf 'No NVIDIA GPU is visible in opd-slime.\n' >&2
  exit 1
fi

actor_gpu=${ACTOR_GPU:-0}
if (( gpu_count == 1 )); then
  math_gpu=${MATH_TEACHER_GPU:-0}
  code_gpu=${CODE_TEACHER_GPU:-0}
  teacher_mem_fraction=${TEACHER_MEM_FRACTION:-0.18}
  rollout_mem_fraction=${ROLLOUT_MEM_FRACTION:-0.22}
elif (( gpu_count == 2 )); then
  math_gpu=${MATH_TEACHER_GPU:-1}
  code_gpu=${CODE_TEACHER_GPU:-1}
  teacher_mem_fraction=${TEACHER_MEM_FRACTION:-0.38}
  rollout_mem_fraction=${ROLLOUT_MEM_FRACTION:-0.50}
else
  math_gpu=${MATH_TEACHER_GPU:-1}
  code_gpu=${CODE_TEACHER_GPU:-2}
  teacher_mem_fraction=${TEACHER_MEM_FRACTION:-0.70}
  rollout_mem_fraction=${ROLLOUT_MEM_FRACTION:-0.50}
fi
for gpu in "$actor_gpu" "$math_gpu" "$code_gpu"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]] || (( gpu >= gpu_count )); then
    printf 'Invalid GPU index %s; %s GPUs are visible.\n' "$gpu" "$gpu_count" >&2
    exit 2
  fi
done

if curl -fsS "http://127.0.0.1:$math_port/health_generate" >/dev/null 2>&1 || \
   { [[ "$teacher_mode" == multi ]] && \
     curl -fsS "http://127.0.0.1:$code_port/health_generate" >/dev/null 2>&1; }; then
  printf 'A teacher health endpoint already occupies port %s or %s.\n' "$math_port" "$code_port" >&2
  exit 1
fi

math_pid=
code_pid=
sampler_pid=
cleanup() {
  [[ -z "$sampler_pid" ]] || kill "$sampler_pid" 2>/dev/null || true
  [[ -z "$math_pid" ]] || kill "$math_pid" 2>/dev/null || true
  [[ -z "$code_pid" ]] || kill "$code_pid" 2>/dev/null || true
  CUDA_VISIBLE_DEVICES="$actor_gpu" ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

wait_for_teacher() {
  local name=$1 pid=$2 port=$3 log_path=$4
  for _ in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:$port/health_generate" >/dev/null 2>&1; then
      return
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      printf '%s teacher exited before becoming ready.\n' "$name" >&2
      tail -100 "$log_path" >&2
      exit 1
    fi
    sleep 1
  done
  printf 'Timed out waiting for %s teacher on port %s.\n' "$name" "$port" >&2
  tail -100 "$log_path" >&2
  exit 1
}

cd /workspace/slime
export HF_HOME="$bench_root/cache/huggingface"
export PYTHONUNBUFFERED=1
export MOPD_TEACHERS_JSON='[{"name":"math_teacher","domain":"math"},{"name":"code_teacher","domain":"code"}]'
if [[ "$teacher_mode" == single ]]; then
  export MOPD_TEACHER_URLS="{\"math\":\"http://127.0.0.1:$math_port/generate\",\"code\":\"http://127.0.0.1:$math_port/generate\"}"
else
  export MOPD_TEACHER_URLS="{\"math\":\"http://127.0.0.1:$math_port/generate\",\"code\":\"http://127.0.0.1:$code_port/generate\"}"
fi

process_start_epoch=$(date +%s)
"$artifact_root/gpu_sampler.sh" "$run_dir/gpu.csv" &
sampler_pid=$!

CUDA_VISIBLE_DEVICES="$math_gpu" python3 -m sglang.launch_server \
  --model-path "$model_root/math-teacher" \
  --host 0.0.0.0 --port "$math_port" --tp 1 \
  --chunked-prefill-size 4096 --mem-fraction-static "$teacher_mem_fraction" \
  --disable-cuda-graph > "$run_dir/math-teacher.log" 2>&1 &
math_pid=$!
wait_for_teacher math "$math_pid" "$math_port" "$run_dir/math-teacher.log"

if [[ "$teacher_mode" == multi ]]; then
  CUDA_VISIBLE_DEVICES="$code_gpu" python3 -m sglang.launch_server \
    --model-path "$model_root/code-teacher" \
    --host 0.0.0.0 --port "$code_port" --tp 1 \
    --chunked-prefill-size 4096 --mem-fraction-static "$teacher_mem_fraction" \
    --disable-cuda-graph > "$run_dir/code-teacher.log" 2>&1 &
  code_pid=$!
  wait_for_teacher code "$code_pid" "$code_port" "$run_dir/code-teacher.log"
fi
teachers_ready_epoch=$(date +%s)

code_gpu_label=$code_gpu
if [[ "$teacher_mode" == single ]]; then
  code_gpu_label=none
fi
printf '%s\n' \
  "gpu_count=$gpu_count" "actor_gpu=$actor_gpu" \
  "math_teacher_gpu=$math_gpu" "code_teacher_gpu=$code_gpu_label" \
  "teacher_mem_fraction=$teacher_mem_fraction" \
  "rollout_mem_fraction=$rollout_mem_fraction" > "$run_dir/layout.env"

export CUDA_VISIBLE_DEVICES="$actor_gpu"
ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port=8265 > "$run_dir/ray.log" 2>&1
ray_ready_epoch=$(date +%s)

source scripts/models/qwen2.5-0.5B.sh
runtime_env_json=$(python3 -c '
import json, os
print(json.dumps({"env_vars": {
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "MOPD_TEACHERS_JSON": os.environ["MOPD_TEACHERS_JSON"],
    "MOPD_TEACHER_URLS": os.environ["MOPD_TEACHER_URLS"],
}}))
')

training_start_epoch=$(date +%s)
set +e
ray job submit --address=http://127.0.0.1:8265 \
  --runtime-env-json="$runtime_env_json" \
  -- python3 train.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node 1 --colocate \
  --hf-checkpoint "$model_root/student" \
  --ref-load "$model_root/student_torch_dist" \
  --save "$run_dir/checkpoints" --save-interval "$save_interval" \
  --prompt-data "$data_root/slime_train.jsonl" \
  --input-key messages --label-key label --apply-chat-template --rollout-shuffle \
  --num-rollout "$steps" --rollout-batch-size 16 --n-samples-per-prompt 4 \
  --rollout-max-prompt-len 1024 --rollout-max-response-len 256 \
  --rollout-temperature 1.0 --rollout-top-p 1.0 \
  --seed 42 --global-batch-size 64 --balance-data \
  --advantage-estimator grpo --use-mopd --mopd-teacher-mode sglang \
  --mopd-teachers "$MOPD_TEACHERS_JSON" --mopd-distill-type token_level \
  --mopd-alpha 0.0 --mopd-eps-low 0.2 --mopd-eps-high 5.0 \
  --mopd-sampling-logprobs-key rollout_log_probs \
  --custom-rm-path slime.rollout.mopd.reward_func \
  --custom-reward-post-process-path slime.rollout.mopd.post_process_rewards \
  --use-kl-loss --kl-loss-coef 0.0 --kl-loss-type low_var_kl \
  --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --clip-grad 1.0 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 --sequence-parallel \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 8192 \
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static "$rollout_mem_fraction" \
  --attention-dropout 0.0 --hidden-dropout 0.0 \
  --bf16 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --attention-backend flash "${MODEL_ARGS[@]}" \
  2>&1 | tee "$run_dir/run.log"
status=${PIPESTATUS[0]}
set -e
end_epoch=$(date +%s)

printf '%s\n' \
  "framework=slime" "teacher_mode=$teacher_mode" "run_id=$run_id" "steps=$steps" \
  "exit_status=$status" "start_epoch=$process_start_epoch" \
  "teachers_ready_epoch=$teachers_ready_epoch" "ray_ready_epoch=$ray_ready_epoch" \
  "training_start_epoch=$training_start_epoch" "end_epoch=$end_epoch" \
  "teacher_start_seconds=$((teachers_ready_epoch - process_start_epoch))" \
  "framework_wall_seconds=$((end_epoch - training_start_epoch))" \
  "wall_seconds=$((end_epoch - process_start_epoch))" > "$run_dir/wall_time.env"

printf '%s\n' "$run_dir"
exit "$status"
