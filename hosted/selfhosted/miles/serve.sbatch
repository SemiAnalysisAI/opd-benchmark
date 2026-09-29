#!/bin/bash
# The Miles Tinker gateway (serve_tinker.py: Megatron LoRA trainer, SGLang samplers) on four 8-GPU Slurm nodes.
#
#   CAMPAIGN=/shared/.../tinker-endpoints sbatch -p PARTITION hosted/selfhosted/miles/serve.sbatch
#
# Layout, matching skyrl/serve.sbatch: the trainer on two nodes, four 4-GPU SGLang engines on the other two, not
# colocated (the gateway forbids --colocate). The Qwen3.6-35B-A3B parallelism is that of Miles'
# scripts/run_qwen3_5_35b_a3b_lora.py: TP2 (two query groups cap it), EP8, full recompute, bshd micro-batches of
# one sequence (Megatron's GatedDeltaNet rejects packed sequences). The gateway rejects a datum longer than
# --max-tokens-per-gpu, so it is 32,768 for 30,720-token responses. Clients see the model as Qwen/Qwen3.6-35B-A3B
# (--tinker-model-name), must not send a LoRA seed or user metadata (the Miles backend drops both), and get at most
# rank 64. Every adapter trains with alpha 32, as on SkyRL.
#
# Inputs under $CAMPAIGN: images/miles-dev-202609290439.sqsh (radixark/miles, Miles 9e4260de with the gateway), hf/.
# Outputs go to $CAMPAIGN/miles/server-<job>: server.json, telemetry/, logs/, checkpoints/. Serves until cancelled.
#SBATCH -J miles-tinker
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=UNLIMITED
#SBATCH -o %x-%j.out
set -euo pipefail
: "${CAMPAIGN:?CAMPAIGN must be set}"
PORT=${PORT:-8000}
IMAGE=${IMAGE:-$CAMPAIGN/images/miles-dev-202609290439.sqsh}
BASE_MODEL=Qwen/Qwen3.6-35B-A3B
MODEL_DIR=$CAMPAIGN/hf/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-32}
ADAPTERS=${ADAPTERS:-6}
source "$CAMPAIGN/opd/hosted/selfhosted/common.sh"
selfhosted_setup miles "$IMAGE" miles-tinker
mkdir -p "$RUN/checkpoints"
selfhosted_ray /opt/sglang/bin

TRAIN_ARGS="--hf-checkpoint $MODEL_DIR --megatron-to-hf-mode bridge \
 --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA --lora-dropout 0.0 --no-gradient-accumulation-fusion \
 --multi-lora-n-adapters $ADAPTERS --target-modules attn,mlp,unembed \
 --tinker-server-port $PORT --tinker-checkpoint-root $RUN/checkpoints --tinker-model-name $BASE_MODEL \
 --optimizer adam --lr 1e-5 --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.999 \
 --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1 \
 --expert-model-parallel-size 8 --expert-tensor-parallel-size 1 \
 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
 --qkv-format bshd --micro-batch-size 1 --max-tokens-per-gpu 32768 \
 --rollout-num-gpus-per-engine 4 --sglang-mem-fraction-static 0.8 --sglang-dtype bfloat16 \
 --sglang-max-lora-rank $LORA_RANK --sglang-lora-backend triton --sglang-context-length 32768 \
 --actor-num-nodes 2 --actor-num-gpus-per-node 8 --rollout-num-gpus 16 \
 --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
 --attention-backend flash --update-weight-buffer-size 536870912"

python3 - "$RUN/server.json" <<PY
import json, sys
json.dump({'framework': 'miles', 'endpoint': 'http://$HEAD_IP:$PORT', 'slurm_job_id': '$SLURM_JOB_ID',
           'nodes': '${NODES[*]}'.split(), 'head': '$HEAD', 'base_model': '$BASE_MODEL', 'model_dir': '$MODEL_DIR',
           'miles_revision': '9e4260de047a704208535c0e90c531929879ab40', 'image': '$IMAGE',
           'lora_rank_cap': $LORA_RANK, 'lora_alpha': $LORA_ALPHA, 'adapter_slots': $ADAPTERS,
           'train_args': '''$TRAIN_ARGS'''.split()}, open(sys.argv[1], 'w'), indent=2)
PY

# serve_tinker.py is an orchestration script on the Ray cluster; the model args come from Miles' registry, as its
# own launcher (miles.utils.external_utils.command_utils.ray_backend) builds them.
srun --overlap -N1 -w "$HEAD" "${CONTAINER[@]}" bash -c "cd /root/miles && \
    export RAY_ADDRESS=$HEAD_IP:6379 MASTER_ADDR=$HEAD_IP PYTHONPATH=/root/Megatron-LM:/root/miles PYTHONUNBUFFERED=1 && \
    MODEL_ARGS=\$(python3 -c 'from miles.utils.external_utils.model_args_utils import shell_safe_model_args; print(shell_safe_model_args(\"qwen3.6-35B-A3B_lora\"))') && \
    echo \"model args: \$MODEL_ARGS\" && \
    eval python3 serve_tinker.py \$MODEL_ARGS $TRAIN_ARGS" > "$RUN/logs/server.log" 2>&1 &
SERVER=$!
wait $SERVER || true
selfhosted_finish
