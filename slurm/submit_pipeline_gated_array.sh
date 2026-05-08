#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/jmooney/society-sim"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/configs/experiments/core.yaml}"
MODELS_PATH="${MODELS_PATH:-$PROJECT_DIR/configs/models.yaml}"
RUN_ROOT="${RUN_ROOT:-/lustre/fs0/scratch/jmooney/latent_profile}"
MANIFEST_DIR="$RUN_ROOT/manifests"
MANIFEST_PATH="${MANIFEST_PATH:-$MANIFEST_DIR/gated_model_ids.txt}"
SBATCH_SCRIPT="$PROJECT_DIR/slurm/run_pipeline_array.sbatch"

mkdir -p \
  "$MANIFEST_DIR" \
  "$RUN_ROOT/logs/slurm" \
  "$RUN_ROOT/logs/array_tasks" \
  "$RUN_ROOT/status" \
  "$RUN_ROOT/tmp"

python3 - <<'PY' "$MODELS_PATH" "$MANIFEST_PATH"
import sys
import yaml

models_path, manifest_path = sys.argv[1], sys.argv[2]
with open(models_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

gated_model_ids = []
for model in config["models"]:
    family = str(model.get("family", ""))
    if family.startswith("llama") or family == "gemma3":
        gated_model_ids.append(str(model["id"]))

with open(manifest_path, "w", encoding="utf-8") as f:
    for model_id in gated_model_ids:
        f.write(model_id + "\n")
PY

COUNT="$(wc -l < "$MANIFEST_PATH" | tr -d ' ')"
if [[ "$COUNT" -eq 0 ]]; then
  echo "Gated manifest is empty: $MANIFEST_PATH" >&2
  exit 2
fi

echo "Submitting gated-only array for $COUNT models using manifest $MANIFEST_PATH"
sbatch --array=0-$((COUNT - 1)) \
  --export=ALL,RUN_ROOT="$RUN_ROOT",CONFIG_PATH="$CONFIG_PATH",MANIFEST_PATH="$MANIFEST_PATH" \
  "$SBATCH_SCRIPT"
