#!/usr/bin/env bash
set -euo pipefail

# Usage: GPU=0 bash scripts/run_orbit.example.sh IMAGE [CONFIG] [OUTPUT]
if [[ $# -lt 1 ]]; then
  printf 'Usage: %s IMAGE [CONFIG] [OUTPUT]\n' "$0" >&2
  exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
image=$1
config=${2:-"$repo_dir/configs/neomap.example.yaml"}
output=${3:-"$repo_dir/outputs/orbit"}

# Image/config/output paths are interpreted from your current working directory.
CUDA_VISIBLE_DEVICES="${GPU:-0}" python "$repo_dir/main_neomap.py" orbit \
  --image "$image" --config "$config" --output-dir "$output" \
  --orbit-angle 30 --radius-scale 1 --num-frames 49
