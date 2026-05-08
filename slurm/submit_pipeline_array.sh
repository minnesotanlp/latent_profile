#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/jmooney/society-sim"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/configs/experiments/core.yaml}"
RUN_ROOT="${RUN_ROOT:-/lustre/fs0/scratch/jmooney/latent_profile}"
MANIFEST_DIR="$RUN_ROOT/manifests"
MANIFEST_PATH="${MANIFEST_PATH:-$MANIFEST_DIR/core_model_ids.txt}"
SBATCH_SCRIPT="$PROJECT_DIR/slurm/run_pipeline_array.sbatch"

mkdir -p \
  "$MANIFEST_DIR" \
  "$RUN_ROOT/logs/slurm" \
  "$RUN_ROOT/logs/array_tasks" \
  "$RUN_ROOT/status" \
  "$RUN_ROOT/tmp"

python3 - <<'PY' "$CONFIG_PATH" "$MANIFEST_PATH"
import sys
import yaml

config_path, manifest_path = sys.argv[1], sys.argv[2]
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model_ids = config["experiment"]["model_ids"]
with open(manifest_path, "w", encoding="utf-8") as f:
    for model_id in model_ids:
        f.write(str(model_id) + "\n")
PY

COUNT="$(wc -l < "$MANIFEST_PATH" | tr -d ' ')"
if [[ "$COUNT" -eq 0 ]]; then
  echo "Manifest is empty: $MANIFEST_PATH" >&2
  exit 2
fi

echo "Submitting array for $COUNT models using manifest $MANIFEST_PATH"
sbatch --array=0-$((COUNT - 1)) \
  --export=ALL,RUN_ROOT="$RUN_ROOT",CONFIG_PATH="$CONFIG_PATH",MANIFEST_PATH="$MANIFEST_PATH" \
  "$SBATCH_SCRIPT"
