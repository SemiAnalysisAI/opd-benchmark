#!/usr/bin/env bash
set -euo pipefail

bench_root=${BENCH_ROOT:-/dev/shm/opd-bench}
artifact_root=${ARTIFACT_ROOT:-$bench_root/artifacts/mopd-b200-full}
output_dir=${OUTPUT_DIR:-$bench_root/results-full/environment}
mkdir -p "$output_dir"

section() {
  printf '\n===== %s =====\n' "$1"
}

{
  section timestamp
  date -u --iso-8601=seconds
  section kernel
  uname -a
  section os-release
  cat /etc/os-release
  section cpu
  lscpu
  section memory
  free -h
  section filesystems
  df -hT
  section mounts
  findmnt -T "$bench_root"
  section gpu-list
  nvidia-smi -L
  section gpu-query
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.memory --format=csv
  section gpu-topology
  nvidia-smi topo -m
  section nvidia-smi
  nvidia-smi
  section docker
  docker version
  section docker-images
  docker image inspect radixark/miles@sha256:37b5eac955caa2104690ac8b55ee2d70579d962546f61b208c6a0392d1df15a6 \
    --format '{{json .RepoDigests}}'
  docker image inspect verlai/verl@sha256:b867883b0dd011363e69ab2ab344922a28c5bd0409e2a324e3ee70fb27ca7543 \
    --format '{{json .RepoDigests}}'
  section apt-build-dependencies
  dpkg-query -W -f='${Package}\t${Version}\n' \
    build-essential python3.12-dev git curl jq
  section repository-revisions
  for repo in prime-rl miles verl; do
    printf '%s\t' "$repo"
    git -C "$bench_root/src/$repo" rev-parse HEAD
    git -C "$bench_root/src/$repo" status --short
  done
  section prime-submodules
  git -C "$bench_root/src/prime-rl" submodule status --recursive
  section model-revisions
  printf '%s\n' \
    'student Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218' \
    'math Qwen/Qwen3-32B 9216db5781bf21249d130ec9da846c4624c16137' \
    'code Qwen/Qwen3-Coder-30B-A3B-Instruct b2cff646eb4bb1d68355c01b18ae02e7cf42d120'
  section tokenizer-hashes
  sha256sum "$bench_root"/models-full/*/tokenizer.json
  section dataset-hashes
  sha256sum "$bench_root"/data-full/*
  section benchmark-script-hashes
  find "$artifact_root" -maxdepth 3 -type f ! -path '*/results/*' ! -path '*/charts/*' \
    -print0 | sort -z | xargs -0 sha256sum
} > "$output_dir/system.txt" 2>&1

"$bench_root/bin/uv" pip freeze \
  --python "$bench_root/src/prime-rl/.venv/bin/python" \
  > "$output_dir/prime-pip-freeze.txt"
docker exec opd-miles-full python3 -m pip freeze \
  > "$output_dir/miles-pip-freeze.txt"
docker exec opd-verl-full python3 -m pip freeze \
  > "$output_dir/verl-pip-freeze.txt"
git -C "$bench_root/src/verl" diff \
  > "$output_dir/verl-working-tree.patch"
cp "$bench_root/data-full/manifest.json" "$output_dir/data-manifest.json"

printf '%s\n' "$output_dir"
