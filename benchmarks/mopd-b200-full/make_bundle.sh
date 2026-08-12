#!/usr/bin/env bash
set -euo pipefail

artifact_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bundle="$artifact_dir/mopd-b200-full-benchmark-bundle.tar.gz"

if find "$artifact_dir" -type f \( \
  -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' -o \
  -name '*.ckpt' \) -print -quit | grep -q .; then
  printf 'Refusing to package model/checkpoint artifacts\n' >&2
  exit 1
fi
if find "$artifact_dir" -type f ! -path "$bundle" -size +100M -print -quit | grep -q .; then
  printf 'Refusing to package a source/evidence file over 100 MB\n' >&2
  exit 1
fi

rm -f -- "$bundle"
tar -C "$(dirname "$artifact_dir")" \
  --exclude='mopd-b200-full/mopd-b200-full-benchmark-bundle.tar.gz' \
  -czf "$bundle" "$(basename "$artifact_dir")"
shasum -a 256 "$bundle"
