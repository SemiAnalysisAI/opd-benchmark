#!/bin/bash
# The SkyRL Tinker API server (Megatron trainer, vLLM samplers) on four 8-GPU Slurm nodes, with telemetry.
#
#   CAMPAIGN=/shared/.../tinker-endpoints sbatch -p PARTITION hosted/selfhosted/skyrl/serve.sbatch
#
# Layout: the Megatron LoRA trainer on two nodes (TP4, EP8), four TP4 vLLM engines on the other two, not colocated,
# so sampling overlaps training. Micro-batches are bin-packed up to 32,768 tokens; one sequence per micro-batch made
# SFT on the short simple_geometry traces 3x slower per token than on caesar_cipher. Every adapter (student and
# teachers) is served by the same engines. The server never expires a session (--session-timeout-sec -1), because
# SkyRL unloads a stale session's adapter and a teacher must stay servable after its SFT client exits. The cluster
# steps (shm cleanup, telemetry, NCCL on InfiniBand, the Ray cluster) are in ../common.sh.
#
# Inputs under $CAMPAIGN: images/skyrl-megatron.sqsh, src/skyrl (the SkyRL checkout, with .venv from `uv sync
# --extra tinker --extra megatron`), hf/ (a Hugging Face cache holding the base model), telemetry-scripts/ (the
# ClusterMAX gpu_monitor.sh, fabric_monitor.sh and system_snapshot.sh). Outputs go to $CAMPAIGN/skyrl/server-<job>:
# server.json (endpoint, revisions, backend config), node-roles.json, telemetry/, logs/. The job serves until cancelled.
#SBATCH -J skyrl-tinker
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=UNLIMITED
#SBATCH -o %x-%j.out
set -euo pipefail
: "${CAMPAIGN:?CAMPAIGN must be set}"
PORT=${PORT:-8000}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.6-35B-A3B}
LORA_ALPHA=${LORA_ALPHA:-32}
SRC=$CAMPAIGN/src/skyrl
source "$CAMPAIGN/opd/hosted/selfhosted/common.sh"
# SkyRL at 7d94ccf crashes serializing a forward_backward metric that is None (importance_ratio:mean with
# loss_fn=importance_sampling); none-metric.patch skips such a metric and logs its name.
PATCH=$SELFHOSTED/skyrl/none-metric.patch
git -C "$SRC" apply --reverse --check "$PATCH" 2>/dev/null || git -C "$SRC" apply "$PATCH"
selfhosted_setup skyrl "$CAMPAIGN/images/skyrl-megatron.sqsh" skyrl-megatron
mkdir -p "$RUN/lora-sync" "$RUN/checkpoints"
export VLLM_USE_FLASHINFER_MOE_FP16=0  # Blackwell: the flashinfer bf16 MoE kernel needs a multiple-of-128 expert size.
export FLA_TILELANG=0                  # Blackwell: the TileLang GatedDeltaNet backward aborts.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
selfhosted_ray "$SRC/.venv/bin"

BACKEND_CONFIG=$(cat <<JSON
{
  "trainer.strategy": "megatron",
  "trainer.placement.colocate_all": false,
  "trainer.placement.policy_num_nodes": 2,
  "trainer.placement.policy_num_gpus_per_node": 8,
  "trainer.policy.language_model_only": true,
  "generator.inference_engine.language_model_only": true,
  "trainer.policy.megatron_config.tensor_model_parallel_size": 4,
  "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
  "trainer.policy.megatron_config.context_parallel_size": 1,
  "trainer.policy.megatron_config.expert_model_parallel_size": 8,
  "trainer.policy.megatron_config.expert_tensor_parallel_size": 1,
  "trainer.policy.megatron_config.lora_config.merge_lora": false,
  "trainer.policy.model.lora.alpha": $LORA_ALPHA,
  "trainer.policy.model.lora.lora_sync_path": "$RUN/lora-sync",
  "trainer.policy.model.lora.max_loras": 4,
  "trainer.policy.model.lora.max_cpu_loras": 8,
  "trainer.max_tokens_per_microbatch": 32768,
  "trainer.max_prompt_length": 2048,
  "generator.sampling_params.max_generate_length": 30720,
  "generator.eval_sampling_params.max_generate_length": 30720,
  "trainer.algorithm.use_kl_loss": false,
  "trainer.policy.optimizer_config.weight_decay": 0.0,
  "generator.inference_engine.num_engines": 4,
  "generator.inference_engine.tensor_parallel_size": 4,
  "generator.inference_engine.backend": "vllm",
  "generator.inference_engine.run_engines_locally": true,
  "generator.inference_engine.distributed_executor_backend": "mp",
  "generator.inference_engine.weight_sync_backend": "nccl",
  "generator.inference_engine.gpu_memory_utilization": 0.85,
  "generator.inference_engine.enforce_eager": false,
  "generator.inference_engine.engine_init_kwargs": {"max_model_len": 32768, "gdn_prefill_backend": "triton",
                                                    "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}},
  "generator.batched": true
}
JSON
)
python3 - "$RUN/server.json" <<PY
import json, sys, subprocess
rev = lambda d: subprocess.run(['git', '-C', d, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
json.dump({'framework': 'skyrl', 'endpoint': 'http://$HEAD_IP:$PORT', 'slurm_job_id': '$SLURM_JOB_ID',
           'nodes': '${NODES[*]}'.split(), 'head': '$HEAD', 'base_model': '$BASE_MODEL', 'skyrl_revision': rev('$SRC'),
           'image': '$CAMPAIGN/images/skyrl-megatron.sqsh', 'lora_alpha': $LORA_ALPHA,
           'backend_config': json.loads('''$BACKEND_CONFIG''')}, open(sys.argv[1], 'w'), indent=2)
PY

# The API server must start as `uv run <flags> -m skyrl.tinker.api`: it rebuilds that command to spawn its engine.
# --frozen --no-sync keeps both on the installed .venv.
cd "$SRC"
srun --overlap -N1 -w "$HEAD" "${CONTAINER[@]}" bash -c "cd $SRC && export RAY_ADDRESS=$HEAD_IP:6379 \
    UV_PROJECT_ENVIRONMENT=$SRC/.venv UV_CACHE_DIR=$CAMPAIGN/uv-cache && \
    uv run --frozen --no-sync --extra tinker --extra megatron -m skyrl.tinker.api \
    --host 0.0.0.0 --port $PORT --base-model $BASE_MODEL --backend megatron \
    --checkpoints-base $RUN/checkpoints --database-url sqlite:///$RUN/tinker.db --session-timeout-sec -1 \
    --backend-config '$BACKEND_CONFIG'" > "$RUN/logs/server.log" 2>&1 &
SERVER=$!
wait $SERVER || true
selfhosted_finish
