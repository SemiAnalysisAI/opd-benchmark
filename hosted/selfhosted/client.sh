#!/bin/bash
# Run one client (smoke.py, sft.py, rl.py, opd.py or mopd.py) against a running self-hosted server, as a step of the
# server's Slurm job on its head node, with the server's endpoint and description in the environment.
#
#   CAMPAIGN=/shared/... hosted/selfhosted/client.sh SERVER_DIR NAME BACKEND -- python hosted/opd/sft.py train ...
#
# SERVER_DIR is the server's output directory (it holds server.json). The client's console goes to
# SERVER_DIR/../runs/NAME.log, and NAME.window.json records its UTC start and end, the server job and the exit
# code, so the server's telemetry can be cut to the run. The step runs in the background and survives the caller.
# CLIENT_VENV (default $CAMPAIGN/client-venv) selects the client environment: SkyRL pins tinker 0.25.0, whose
# responses the 0.30 SDK rejects (it requires sample_sequence_ids), so SkyRL clients use a 0.25.0 environment.
set -euo pipefail
: "${CAMPAIGN:?CAMPAIGN must be set}"
SERVER=$(cd "$1" && pwd); NAME=$2; BACKEND=$3; shift 3
[[ ${1:-} == -- ]] && shift
RUNS=$(dirname "$SERVER")/runs
VENV=${CLIENT_VENV:-$CAMPAIGN/client-venv}
mkdir -p "$RUNS"
read -r JOB HEAD ENDPOINT < <(python3 -c "import json,sys; s=json.load(open(sys.argv[1])); print(s['slurm_job_id'], s['head'], s['endpoint'])" "$SERVER/server.json")
[[ -e $RUNS/$NAME.log ]] && { echo "$RUNS/$NAME.log exists; pick a new name" >&2; exit 1; }
printf -v COMMAND '%q ' "$@"
nohup srun --overlap --jobid="$JOB" -N1 -w "$HEAD" bash -c "
    export HF_HOME=$CAMPAIGN/hf HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PATH=$VENV/bin:\$PATH
    export CAMPAIGN_PYDEPS=$VENV/lib/python3.12/site-packages
    export TINKER_BASE_URL=$ENDPOINT TINKER_SERVER_INFO=$SERVER/server.json
    cd $CAMPAIGN/opd
    start=\$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    $COMMAND; code=\$?
    python3 -c 'import json,sys; json.dump(dict(name=sys.argv[1], backend=sys.argv[2], server=sys.argv[3], slurm_job_id=sys.argv[4],
        start_utc=sys.argv[5], end_utc=sys.argv[6], exit_code=int(sys.argv[7]), command=sys.argv[8]), open(sys.argv[9], \"w\"), indent=2)' \
        $NAME $BACKEND $SERVER $JOB \$start \$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ) \$code '$COMMAND' $RUNS/$NAME.window.json
    exit \$code" > "$RUNS/$NAME.log" 2>&1 &
echo "started $NAME on $HEAD (job $JOB): $RUNS/$NAME.log"
