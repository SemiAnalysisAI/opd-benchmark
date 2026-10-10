#!/bin/bash
# The Fireworks OPD/MOPD campaign: lease with both uploaded teachers, teacher check, MOPD, then OPD
# (after dropping the geometry teacher), then release. Stops at the first failure; the watchdog
# releases everything at the budget or if this script dies.
#
#   [SAMPLER_REPLICAS=5] [SAMPLER_SHAPE=... SAMPLER_GPUS=1] [TEACHER_SHAPE=...] [HOT_LOAD_TRANSITION=ASYNC]
#   [RUN_ARGS=--skip-initial-eval]
#   [MOPD_ONLY=1] hosted/opd/run_fireworks.sh /abs/new-lease-dir BUDGET_USD
#
# Rerunning with the same directory reuses a lease that is already up and a teacher check that passed.
set -u
LEASE=$1; BUDGET=$2
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE/../.."
PY=${PYTHON:-python}
CAESAR=qwen3p6-35b-a3b-caesar-cipher-sft-tinker
GEOMETRY=qwen3p6-35b-a3b-simple-geometry-sft-tinker
export FIREWORKS_LEASE=$LEASE TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
say() { echo "{\"utc\": \"$(date -u +%FT%TZ)\", \"stage\": \"$1\"}"; }
fail() { say "failed: $1"; "$PY" hosted/opd/fireworks_lease.py down --root "$LEASE" > /dev/null; exit 1; }

if [ -f "$LEASE/lease.json" ] && [ ! -f "$LEASE/TRAINING_FINISHED" ]; then
    say lease_reused
else
    say lease_up
    "$PY" -u hosted/opd/fireworks_lease.py up --root "$LEASE" --budget-usd "$BUDGET" --sampler-replicas "${SAMPLER_REPLICAS:-1}" \
        ${SAMPLER_SHAPE:+--sampler-shape "$SAMPLER_SHAPE"} --sampler-gpus "${SAMPLER_GPUS:-2}" \
        ${TEACHER_SHAPE:+--teacher-shape "$TEACHER_SHAPE"} --hot-load-transition "${HOT_LOAD_TRANSITION:-SYNC}" \
        --teacher caesar_cipher=$CAESAR --teacher simple_geometry=$GEOMETRY > "$LEASE.lease.log" 2>&1 &
    until [ -f "$LEASE/teachers.json" ] && grep -q '"ready"' "$LEASE/lease-events.jsonl" 2>/dev/null; do
        kill -0 $! 2>/dev/null || fail lease_up; sleep 15
    done
fi
RUNS=$LEASE-runs; mkdir -p "$RUNS"

if grep -q '"passed": true' "$LEASE/teacher-check.json" 2>/dev/null; then
    say teacher_check_reused
else
    say teacher_check
    "$PY" -u hosted/opd/check_teachers.py --backend fireworks --teachers "$LEASE/teachers.json" \
        --rollouts hosted/opd/reference/teacher-check-rollouts.json --output "$LEASE/teacher-check.json" \
        > "$RUNS/teacher-check.log" 2>&1 || fail teacher_check
fi

say mopd
"$PY" -u hosted/opd/mopd.py --backend fireworks --teachers "$LEASE/teachers.json" --output "$RUNS/mopd" ${RUN_ARGS:-} \
    > "$RUNS/mopd.log" 2>&1 || fail mopd

if [ -n "${MOPD_ONLY:-}" ]; then
    say release
    "$PY" hosted/opd/fireworks_lease.py down --root "$LEASE" > /dev/null
    say complete
    exit 0
fi

say drop_geometry_teacher
"$PY" hosted/opd/fireworks_lease.py drop-teacher --root "$LEASE" --domain simple_geometry || fail drop_teacher

say opd
CAESAR_TEACHER=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['caesar_cipher'])" "$LEASE/teachers.json")
"$PY" -u hosted/opd/opd.py --backend fireworks --domain caesar_cipher --teacher "$CAESAR_TEACHER" --output "$RUNS/opd-caesar" ${RUN_ARGS:-} \
    > "$RUNS/opd-caesar.log" 2>&1 || fail opd

say release
"$PY" hosted/opd/fireworks_lease.py down --root "$LEASE" > /dev/null
say complete
