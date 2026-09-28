#!/bin/bash
# MOPD with a serverless Qwen3.8-27B student and the uploaded Qwen3.6 SFT teachers: a teacher-only lease
# (one validated 1xB200 deployment per teacher, under the budget watchdog), the MOPD run on the serverless
# Training API, then release. Stops at the first failure; the watchdog releases the teachers at the budget
# or if this script dies. Serverless token spend is per token and is not covered by the watchdog.
#
#   [RUN_ARGS="--max-tokens 16000 --eval-every 20"] hosted/opd/run_serverless.sh /abs/new-lease-dir TEACHER_BUDGET_USD
set -u
LEASE=$1; BUDGET=$2
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE/../.."
PY=${PYTHON:-python}
SHAPE=accounts/fireworks/deploymentShapes/rft-qwen3p6-35b-a3b-rl-b200-bf16-w1-p1/versions/a35zwd97
export FIREWORKS_LEASE=$LEASE TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
say() { echo "{\"utc\": \"$(date -u +%FT%TZ)\", \"stage\": \"$1\"}"; }
fail() { say "failed: $1"; "$PY" hosted/opd/fireworks_lease.py down --root "$LEASE" > /dev/null; exit 1; }

say teachers_up
"$PY" -u hosted/opd/fireworks_lease.py up --root "$LEASE" --budget-usd "$BUDGET" --teachers-only --teacher-shape $SHAPE \
    --teacher caesar_cipher=qwen3p6-35b-a3b-caesar-cipher-sft-tinker \
    --teacher simple_geometry=qwen3p6-35b-a3b-simple-geometry-sft-tinker > "$LEASE.lease.log" 2>&1 &
until [ -f "$LEASE/teachers.json" ] && grep -q '"ready"' "$LEASE/lease-events.jsonl" 2>/dev/null; do
    kill -0 $! 2>/dev/null || fail teachers_up; sleep 15
done
RUNS=$LEASE-runs; mkdir -p "$RUNS"

say mopd
"$PY" -u hosted/opd/mopd.py --backend fireworks-serverless --student-model qwen3.8-27b --teacher-model qwen3.6-35b-a3b \
    --teachers "$LEASE/teachers.json" --output "$RUNS/mopd" ${RUN_ARGS:-} > "$RUNS/mopd.log" 2>&1 || fail mopd

say release
"$PY" hosted/opd/fireworks_lease.py down --root "$LEASE" > /dev/null
say complete
