#!/usr/bin/env bash
# Reproduce the caesar_cipher OPD benchmark end to end: Qwen3.6-35B-A3B student, the caesar_cipher
# GRPO teacher, the shared recipe (20 updates, thinking on, 32k context).
#
#   tools/reproduce.sh all                 reference benchmarks, every framework, then the report
#   tools/reproduce.sh miles verl          chosen frameworks (miles, nemo-rl, prime-rl, slime, verl)
#   tools/reproduce.sh references report   base and teacher benchmarks; telemetry report over finished runs
#
# For each framework: assets -> prepare -> runtime -> run (blocks until the Slurm job ends) -> export
# Hugging Face weights of every saved checkpoint -> benchmark each (the final one as <run>, earlier ones
# as <run>-step<N>). Steps whose output already exists are skipped, so a re-run resumes where the last
# one stopped. Run it on a Slurm login node with Pyxis/Enroot.
#
# Settings (environment):
#   PARTITION       Slurm partition (required)
#   SITE            site file for the two-node frameworks (default config/site.local.json)
#   NEMO_RL_SITE    site file for NeMo-RL, which needs a third node (default config/site-nemo-rl.local.json)
#   RUNS            campaigns and benchmark outputs (default /shared/opd-runs)
#   NAME            campaign suffix: campaigns are $RUNS/<framework>-$NAME (default caesar)
#   RUNTIME         container images, tool venv, caches, logs (default /shared/opd-runtime)
#   GPU_NODE        node for single-node jobs (default: the site's trainer node)
#   BENCH_CAMPAIGN  verl campaign whose vLLM venv runs the benchmarks (default: $RUNS/verl-$NAME, built if needed)
#   DEADLOCK_MIN    NeMo-RL: cancel the run if its first weight sync has not finished after this many
#                   minutes (default 20); a hung run otherwise holds 24 GPUs until its time limit
set -euo pipefail
[[ $# -gt 0 ]] || { awk 'NR > 1 && !/^#/ {exit} NR > 1' "$0"; exit 2; }

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${PARTITION:?Set PARTITION to your Slurm partition}"
SITE=${SITE:-$REPO/config/site.local.json}
NEMO_RL_SITE=${NEMO_RL_SITE:-$REPO/config/site-nemo-rl.local.json}
RUNS=${RUNS:-/shared/opd-runs}
NAME=${NAME:-caesar}
RUNTIME=${RUNTIME:-/shared/opd-runtime}
DEADLOCK_MIN=${DEADLOCK_MIN:-20}
DOMAIN=caesar_cipher
MILES_IMAGE='docker://radixark/miles@sha256:59a11219eae0defc6594ec678fafe4e897c16904263223f79968cd3e0209a502'
NEMO_RL_IMAGE='docker://nvcr.io#nvidia/nemo-rl:v0.7.0'
TOOLS=$RUNTIME/tools-venv
export UV_CACHE_DIR=$RUNTIME/uv-cache UV_PYTHON_INSTALL_DIR=$RUNTIME/python UV_LINK_MODE=copy
# Slurm sets NVIDIA_VISIBLE_DEVICES=void for job steps; Pyxis then exposes no GPUs or driver libraries.
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
mkdir -p "$RUNS/benchmarks" "$RUNTIME/logs"

say() { echo "[$(date -u +%T)] $*"; }
site() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"; }
recipe() { python3 -c "import sys; sys.path.insert(0, '$REPO'); from shared import recipe; print($1)"; }
node() { echo "${GPU_NODE:-$(site "$SITE" trainer_node)}"; }
on_node() {  # on_node GPUS [srun args...] -- command...: run one exclusive step on GPU_NODE
  local gpus=$1 args=(); shift
  while [[ $1 != -- ]]; do args+=("$1"); shift; done; shift
  srun -p "$PARTITION" -N1 -n1 -w "$(node)" --gres=gpu:"$gpus" --exclusive --time=04:00:00 "${args[@]}" bash -c "$*"
}
in_image() {  # in_image IMAGE GPUS -- command...: on_node in a writable container with /shared mounted
  local image=$1 gpus=$2; shift 3
  on_node "$gpus" --container-image="$image" --container-writable --container-remap-root \
    --no-container-mount-home --container-mounts=/shared:/shared -- "$@"
}
campaign() { echo "$RUNS/$1-$NAME"; }
latest_run() { python3 -c "import json; print(json.load(open('$(campaign "$1")/active-run.json'))['result'])"; }
run_ok() { [[ -f $(campaign "$1")/active-run.json ]] && [[ $(cat "$(latest_run "$1")/exit-code.txt" 2>/dev/null) == 0 ]]; }

tools() {  # pinned uv and Hugging Face CLI; links the site's uv path to it if absent
  [[ -x $TOOLS/bin/uv ]] || { python3 -m venv "$TOOLS"; "$TOOLS/bin/pip" install -q uv==0.11.33 'huggingface_hub[cli]==0.35.3'; }
  local uv; uv=$(site "$SITE" uv)
  [[ -e $uv ]] || { mkdir -p "$(dirname "$uv")"; ln -s "$TOOLS/bin/uv" "$uv"; }
}

assets() {  # download the pinned base and teacher, then fuse the teacher into the base layout
  local base teachers hub
  base=$(site "$SITE" base_model); teachers=$(site "$SITE" teachers); hub=$(site "$SITE" assets)/hub
  [[ -f $base/config.json ]] || { say "downloading the base model"; "$TOOLS/bin/hf" download "$(recipe 'recipe.BASE_MODEL[0]')" \
    --revision "$(recipe 'recipe.BASE_MODEL[1]')" --local-dir "$base" >/dev/null; }
  [[ -f $hub/$DOMAIN/config.json ]] || { say "downloading the $DOMAIN teacher"; "$TOOLS/bin/hf" download \
    "$(recipe "recipe.TEACHERS['$DOMAIN'][0]")" --revision "$(recipe "recipe.TEACHERS['$DOMAIN'][1]")" \
    --local-dir "$hub/$DOMAIN" >/dev/null; }
  [[ -f $teachers/$DOMAIN/config.json ]] || { image miles; say "fusing the teacher"; \
    in_image "$(site "$SITE" container_image)" 1 -- python3 "$REPO/tools/fuse_teacher.py" "$hub/$DOMAIN" "$teachers/$DOMAIN" "$base"; }
}

image() {  # image miles|nemo-rl: import the digest- or tag-pinned container once
  local path from
  if [[ $1 == nemo-rl ]]; then path=$(site "$NEMO_RL_SITE" nemo_rl_container); from=$NEMO_RL_IMAGE
  else path=$(site "$SITE" container_image); from=$MILES_IMAGE; fi
  [[ -f $path ]] && return
  say "importing $from"
  on_node 0 -- "ENROOT_TEMP_PATH=/tmp enroot import -o $path.part '$from' && mv $path.part $path"
}

megatron_checkpoint() {  # Miles and Slime: convert the base model to Megatron torch_dist with Miles' converter
  local out; out=$(site "$SITE" megatron_model)
  [[ -f $out/latest_checkpointed_iteration.txt ]] && return
  prepare miles
  say "converting the base model to torch_dist"
  in_image "$(site "$SITE" container_image)" 8 -- "set -euo pipefail; cd $(campaign miles)/source
    read -ra args <<< \"\$(python3 miles/utils/external_utils/model_args_utils.py qwen3.6-35B-A3B)\"
    PYTHONPATH=$(site "$SITE" megatron_source):\$PWD torchrun --nproc-per-node 8 tools/convert_hf_to_torch_dist.py \
      \"\${args[@]}\" --hf-checkpoint $(site "$SITE" base_model) --save $out --mtp-num-layers 1"
}

prepare() {  # create the single-task campaign with tools/prepare.py
  local fw=$1 site=$SITE
  [[ $fw == nemo-rl ]] && site=$NEMO_RL_SITE
  [[ -d $(campaign "$fw") ]] && return
  say "preparing $(campaign "$fw")"
  python3 "$REPO/tools/prepare.py" "$fw" --site "$site" --output "$(campaign "$fw")" --domains "$DOMAIN"
}

flashinfer() {  # install prebuilt FlashInfer kernels matching the venv's flashinfer-python version
  local py=$1 uv v; uv=$(site "$SITE" uv)
  v=$("$uv" pip list --python "$py" 2>/dev/null | awk '$1 == "flashinfer-python" {print $2}')
  "$uv" pip list --python "$py" 2>/dev/null | grep -q '^flashinfer-jit-cache ' && return
  "$uv" pip install -q --python "$py" --no-deps --index-url https://flashinfer.ai/whl/ "flashinfer-cubin==$v"
  "$uv" pip install -q --python "$py" --no-deps --index-url https://flashinfer.ai/whl/cu130/ "flashinfer-jit-cache==$v+cu130"
}

runtime() {  # build what the campaign needs before submission (see docs/SETUP.md)
  local fw=$1 c uv; c=$(campaign "$fw"); uv=$(site "$SITE" uv)
  case $fw in
  miles|slime)
    image miles; megatron_checkpoint
    # Install reasoning-gym with the container's Python 3.12: its pycosat dependency has no cp312 wheel.
    [[ -d $c/pydeps/reasoning_gym ]] || in_image "$(site "$SITE" container_image)" 0 -- \
      "python3 -m pip install -q --target $c/pydeps reasoning-gym==0.1.25"
    if [[ $fw == slime ]] && ! (cd "$c" && sha256sum -c --quiet wheels.sha256 2>/dev/null); then
      "$TOOLS/bin/pip" download -q --only-binary=:all: --no-deps --platform manylinux2014_x86_64 --python-version 312 \
        --implementation cp --abi cp312 --dest "$c/wheels" numpy==1.26.4 scipy==1.15.3
      (cd "$c" && sha256sum -c wheels.sha256)
    fi ;;
  prime-rl|verl)
    local extras=(--all-extras)
    [[ $fw == verl ]] && extras=(--all-packages --extra vllm --extra fsdp)
    [[ -x $c/source/.venv/bin/python ]] || { say "building the $fw venv"; (cd "$c/source" && "$uv" sync --frozen --python 3.12 "${extras[@]}"); }
    [[ $fw == prime-rl ]] && "$uv" pip install -q --python "$c/source/.venv/bin/python" --no-deps -e "$c/environments/rg_tasks"
    flashinfer "$c/source/.venv/bin/python"
    [[ -d $c/pydeps/reasoning_gym ]] || "$uv" pip install -q --target "$c/pydeps" --python "$c/source/.venv/bin/python" reasoning-gym==0.1.25 ;;
  nemo-rl)
    image nemo-rl ;;
  esac
}

nemo_rl_watchdog() {  # cancel a NeMo-RL run whose first trainer->vLLM weight sync never completes
  # That sync precedes the first update (async_grpo_train, grpo.py:3551-3554 at the pinned revision).
  # Every Qwen3.6-35B-A3B run we tried hung there while rollouts kept going in the background.
  local c previous result log started; c=$(campaign nemo-rl); previous=$1
  # The controller records the new run in active-run.json once Slurm grants the allocation.
  until [[ -f $c/active-run.json ]] && [[ $(latest_run nemo-rl) != "$previous" ]]; do sleep 20; done
  result=$(latest_run nemo-rl); log=
  while [[ ! -f $result/end.json ]]; do
    log=$(ls "$result"/*-logs/ray-driver.log 2>/dev/null | head -n 1)
    if [[ -n $log ]] && grep -aq "Refitting policy generation" "$log"; then
      started=${started:-$(date +%s)}
      grep -aq "refit completed successfully" "$log" && return
      if (( $(date +%s) - started > DEADLOCK_MIN * 60 )); then
        { echo "first weight sync did not complete in $DEADLOCK_MIN min"
          grep -a -o "Error in VllmInternalWorkerExtension[^[]*\|Worker failed to update weights[^[]*" "$log" | sort | uniq -c
        } | tee "$result/DEADLOCK"
        scancel "$(python3 -c "import json; print(json.load(open('$c/active-run.json'))['job_id'])")"
        return
      fi
    fi
    sleep 30
  done
}

run() {  # submit the campaign and wait for its Slurm allocation to end
  local fw=$1 c; c=$(campaign "$fw")
  run_ok "$fw" && { say "$fw: already finished ($(latest_run "$fw"))"; return; }
  say "$fw: submitting"
  [[ $fw == nemo-rl ]] && nemo_rl_watchdog "$( [[ -f $c/active-run.json ]] && latest_run nemo-rl )" &
  python3 "$c/launch.py" "$c" --partition "$PARTITION" --submit || true
  wait
  run_ok "$fw" || { say "$fw: run $(latest_run "$fw") did not finish cleanly; see its results directory"; return 1; }
}

export_hf() {  # export_hf FRAMEWORK STEP: print the path of the Hugging Face weights saved after STEP updates,
  # converting them first if the framework saves another format. Fails if there is no checkpoint at STEP.
  local fw=$1 n=$2 c result name; c=$(campaign "$fw"); result=$(latest_run "$fw"); name=$(basename "$result")
  case $fw in
  miles|slime) [[ -d $c/checkpoints/$name/hf/iter_$(( n - 1 )) ]] && echo "$c/checkpoints/$name/hf/iter_$(( n - 1 ))" ;;
  prime-rl)
    local step=$c/checkpoints/$name/checkpoints/step_$n  # Prime-RL nests checkpoints/
    [[ -d $step ]] || return 1
    [[ -f $step/weights/config.json ]] || on_node 8 -- "cd $c/source && \
      PRL_ATTEMPT_CONFIG_DIR=$(python3 -c "import json; print(json.load(open('$result/resolved-paths.json'))['config'])") \
      $(site "$SITE" uv) run --project $c/source --no-sync torchrun --nproc-per-node 8 tools/convert_dcp_to_bf16.py $step $step/weights" >&2
    echo "$step/weights" ;;
  verl)
    local actor=$c/checkpoints/$name/global_step_$n/actor
    [[ -d $actor ]] || return 1
    [[ -f $actor/huggingface-merged/config.json ]] || on_node 0 -- "cd $c/source && CUDA_VISIBLE_DEVICES= .venv/bin/python -m \
      verl.model_merger merge --backend fsdp --local_dir $actor --target_dir $actor/huggingface-merged --trust-remote-code" >&2
    echo "$actor/huggingface-merged" ;;
  nemo-rl)
    # Follows NeMo-RL's README (examples/converters/convert_megatron_to_hf.py). Untested: no run has saved a checkpoint yet.
    local step=$c/checkpoints/$name/step_$n
    [[ -d $step ]] || return 1
    [[ -f $step/hf/config.json ]] || in_image "$(site "$NEMO_RL_SITE" nemo_rl_container)" 8 -- "cd /opt/nemo-rl && \
      uv run --extra mcore python examples/converters/convert_megatron_to_hf.py --config $step/config.yaml \
      --hf-model-name $(site "$NEMO_RL_SITE" base_model) --megatron-ckpt-path $(ls -d "$step"/policy/weights/iter_* | tail -n 1) \
      --hf-ckpt-path $step/hf" >&2
    echo "$step/hf" ;;
  esac
}

benchmark() {  # benchmark NAME MODEL: run tools/benchmark.py on one node with the verl campaign's vLLM venv
  local name=$1 model=$2 out=$RUNS/benchmarks/$1 v=${BENCH_CAMPAIGN:-$(campaign verl)}
  [[ -f $out/summary.json ]] && { say "benchmark $name: done"; return; }
  [[ -n ${BENCH_CAMPAIGN:-} ]] || { prepare verl; runtime verl; }
  rm -rf "$out"
  say "benchmarking $name"
  on_node 8 -- "unset ROCR_VISIBLE_DEVICES; export CAMPAIGN_PYDEPS=$v/pydeps HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 \
    VLLM_CACHE_ROOT=$v/runtime-cache/vllm FLASHINFER_WORKSPACE_BASE=$v/runtime-cache TRITON_CACHE_DIR=$v/runtime-cache/triton; \
    $v/source/.venv/bin/python $REPO/tools/benchmark.py $model $out --domains caesar_cipher simple_geometry"
}

framework() {
  local fw=$1
  say "==== $fw"
  tools; assets; prepare "$fw"; runtime "$fw"; run "$fw"
  local name bench model n last; name=$(basename "$(latest_run "$fw")"); last=$(recipe recipe.UPDATES)
  for n in $(recipe "' '.join(map(str, range(recipe.SAVE_INTERVAL, recipe.UPDATES + 1, recipe.SAVE_INTERVAL)))"); do
    bench=$name; (( n == last )) || bench=$name-step$n
    [[ -f $RUNS/benchmarks/$bench/summary.json ]] && { say "benchmark $bench: done"; continue; }
    model=$(export_hf "$fw" "$n") || { say "$fw: no checkpoint after $n updates; skipping its benchmark"; continue; }
    benchmark "$bench" "$model"
  done
}

references() {
  tools; assets
  benchmark base "$(site "$SITE" base_model)"
  benchmark teacher-caesar "$(site "$SITE" teachers)/$DOMAIN"
}

report() {
  local results=()
  for fw in miles nemo-rl prime-rl slime verl; do
    [[ -f $(campaign "$fw")/active-run.json ]] && results+=("$(latest_run "$fw")")
  done
  python3 "$REPO/tools/report.py" "${results[@]}" --output "$RUNS/benchmarks/telemetry.json" >/dev/null
  say "report: $RUNS/benchmarks/telemetry.json; benchmarks: $RUNS/benchmarks/*/summary.json"
}

[[ $1 == all ]] && set -- references miles prime-rl slime verl nemo-rl report
for target in "$@"; do
  case $target in
  miles|prime-rl|slime|verl|nemo-rl) framework "$target" || say "$target failed; continuing" ;;
  references) references ;;
  report) report ;;
  *) echo "Unknown target: $target"; exit 2 ;;
  esac
done
