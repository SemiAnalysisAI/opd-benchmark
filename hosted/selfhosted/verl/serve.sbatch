#!/bin/bash
# The verl-tinker server (verl-recipe verl_tinker: FSDP LoRA actor, colocated vLLM rollout, optional frozen vLLM
# teachers) on four 8-GPU Slurm nodes, with telemetry.
#
#   CAMPAIGN=/shared/... CONFIG=actor_rollout.yaml sbatch -p PARTITION hosted/selfhosted/verl/serve.sbatch
#
# verl-tinker holds one trainable model, so each training client (every SFT teacher, RL, MOPD) gets its own server
# job. CONFIG is a file in this directory: actor_rollout.yaml (SFT, RL) or mopd_teachers.yaml (MOPD, with the
# merged SFT teachers from merge_teacher.py). The runtime is verl-recipe's own `uv sync --project verl_tinker`
# environment in $CAMPAIGN/src/verl-recipe/verl_tinker/.venv, run inside the CUDA 13 SkyRL image. The server
# speaks JSON only, so clients use tinker 0.19 (the version it pins); 0.23 and later send protobuf.
# Outputs go to $CAMPAIGN/verl/server-<job>: server.json, telemetry/, logs/, checkpoints/. Serves until cancelled.
#SBATCH -J verl-tinker
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=UNLIMITED
#SBATCH -o %x-%j.out
set -euo pipefail
: "${CAMPAIGN:?CAMPAIGN must be set}"
: "${CONFIG:=actor_rollout.yaml}"
PORT=${PORT:-8000}
SRC=$CAMPAIGN/src/verl-recipe
VENV=$SRC/verl_tinker/.venv
source "$CAMPAIGN/opd/hosted/selfhosted/common.sh"
selfhosted_setup verl "$CAMPAIGN/images/skyrl-megatron.sqsh" skyrl-megatron
mkdir -p "$RUN/checkpoints"
cp "$SELFHOSTED/verl/$CONFIG" "$RUN/config.yaml"
export TINKER_SERVER_MODEL_NAME=Qwen/Qwen3.6-35B-A3B TINKER_SERVER_PORT=$PORT TINKER_CHECKPOINT_DIR=$RUN/checkpoints
export TINKER_SERVER_MODEL_PATH=$CAMPAIGN/hf/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
export TINKER_TEACHERS_DIR=$CAMPAIGN/verl/teachers VLLM_USE_FLASHINFER_MOE_FP16=0 FLA_TILELANG=0
selfhosted_ray "$VENV/bin"

python3 - "$RUN/server.json" <<PY
import json, sys, subprocess
rev = lambda d: subprocess.run(['git', '-C', d, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
json.dump({'framework': 'verl', 'endpoint': 'http://$HEAD_IP:$PORT', 'slurm_job_id': '$SLURM_JOB_ID',
           'nodes': '${NODES[*]}'.split(), 'head': '$HEAD', 'base_model': '$TINKER_SERVER_MODEL_NAME',
           'verl_recipe_revision': rev('$SRC'), 'config': '$CONFIG', 'image': '$CAMPAIGN/images/skyrl-megatron.sqsh',
           'node_roles': {'shared': '${NODES[*]}'.split()} if '$CONFIG' == 'actor_rollout.yaml' else None},
          open(sys.argv[1], 'w'), indent=2)
PY

srun --overlap -N1 -w "$HEAD" "${CONTAINER[@]}" bash -c "cd $SRC/verl_tinker && export RAY_ADDRESS=$HEAD_IP:6379 && \
    $VENV/bin/python -m verl_tinker.start --config $RUN/config.yaml" > "$RUN/logs/server.log" 2>&1 &
SERVER=$!
for _ in $(seq 180); do  # Readiness is healthz reporting ready, not an open port (actor and rollout load after it).
    curl -sf "http://$HEAD_IP:$PORT/api/v1/healthz" | grep -q ready && { echo "ready $(date -u +%FT%TZ)" > "$RUN/ready"; break; }
    kill -0 $SERVER 2>/dev/null || break
    sleep 20
done
wait $SERVER || true
selfhosted_finish
