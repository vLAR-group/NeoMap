#!/usr/bin/env bash
set -euo pipefail

# Dataset reproduction helper. Full benchmark preparation is documented later.
# Usage: GPU=0 bash run_bash/run_dataset.example.sh IMAGE_DIR SPARSE_DIR [CONFIG] [OUTPUT]
if [[ $# -lt 2 ]]; then
  printf 'Usage: %s IMAGE_DIR SPARSE_DIR [CONFIG] [OUTPUT]\n' "$0" >&2
  exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
image_dir=$1
sparse_dir=$2
config=${3:-"$repo_dir/configs/neomap.example.yaml"}
output=${4:-outputs/dataset_scene}

# The dataset renderer needs its additional geometry/COLMAP dependencies.
# Do not add another project's machine-specific paths to this public template.
CUDA_VISIBLE_DEVICES="${GPU:-0}" python "$repo_dir/main_static_wan22_inner_loop.py" \
  --image_dir "$image_dir" --sparse_dir "$sparse_dir" \
  --config "$config" --output_dir "$output" \
  --num_frames 49 --task i2v-A14B --size '1280*720' --convert_model_dtype
