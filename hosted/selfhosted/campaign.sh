#!/bin/bash
# Run the rest of one framework's campaign on a running server, one stage at a time, then summarize every run.
#
#   CAMPAIGN=/shared/... CLIENT_VENV=... hosted/selfhosted/campaign.sh FRAMEWORK SERVER_DIR [STAGE ...]
#
# Stages (default: sft rl mopd), each run through client.sh so it gets its own telemetry window:
#   sft   both SFT teachers at once (sft.py train, the released-teacher defaults); skipped if they already exist
#   rl    GRPO on both domains (rl.py defaults)
#   mopd  MOPD from the two SFT teachers (mopd.py defaults); teachers.json is built from the teachers' teacher.json
# A stage waits for the previous one and stops the campaign if it fails. Outputs: $CAMPAIGN/FRAMEWORK/out/<run>,
# $CAMPAIGN/FRAMEWORK/runs/<run>.{log,window.json,summary.json}.
set -euo pipefail
: "${CAMPAIGN:?CAMPAIGN must be set}"
FRAMEWORK=$1; SERVER=$(cd "$2" && pwd); shift 2
if [[ $# -eq 0 ]]; then STAGES=(sft rl mopd); else STAGES=("$@"); fi
OUT=$CAMPAIGN/$FRAMEWORK/out; RUNS=$CAMPAIGN/$FRAMEWORK/runs
SELFHOSTED=$CAMPAIGN/opd/hosted/selfhosted
mkdir -p "$OUT" "$RUNS"
log() { echo "$(date -u +%FT%TZ) $*"; }

wait_run() {  # Wait for a client run's window file; fail if it exited non-zero.
    local name=$1
    until [[ -f $RUNS/$name.window.json ]]; do sleep 30; done
    local code; code=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['exit_code'])" "$RUNS/$name.window.json")
    "$CAMPAIGN/client-venv/bin/python" "$SELFHOSTED/summarize.py" "$RUNS/$name.window.json" "$OUT/$name" \
        --telemetry-code "$CAMPAIGN/telemetry-scripts" || log "summary failed for $name"
    [[ $code == 0 ]] || { log "$name exited $code; stopping"; exit 1; }
    log "$name complete"
}

start() {  # start NAME -- COMMAND...
    local name=$1; shift 2
    [[ -f $RUNS/$name.window.json ]] && { log "$name already ran"; return; }
    [[ -f $RUNS/$name.log ]] || bash "$SELFHOSTED/client.sh" "$SERVER" "$name" "$FRAMEWORK" -- "$@"
}

for stage in "${STAGES[@]}"; do
    case $stage in
    sft)
        for d in caesar_cipher simple_geometry; do
            start "sft-$d" -- python hosted/opd/sft.py train --backend "$FRAMEWORK" --domain "$d" --output "$OUT/sft-$d"
        done
        wait_run sft-caesar_cipher; wait_run sft-simple_geometry ;;
    rl)
        start rl -- python hosted/opd/rl.py --backend "$FRAMEWORK" --output "$OUT/rl"
        wait_run rl ;;
    mopd)
        python3 - "$OUT" > "$OUT/teachers.json" <<'PY'
import json, sys
out = sys.argv[1]
print(json.dumps({d: json.load(open(f'{out}/sft-{d}/teacher.json'))['sampler_path'] for d in ('caesar_cipher', 'simple_geometry')}, indent=2))
PY
        start mopd -- python hosted/opd/mopd.py --backend "$FRAMEWORK" --teachers "$OUT/teachers.json" --output "$OUT/mopd"
        wait_run mopd ;;
    *) log "unknown stage $stage"; exit 2 ;;
    esac
done
log "campaign stages done: ${STAGES[*]}"
