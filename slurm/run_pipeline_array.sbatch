#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=latent_profile_models
#SBATCH --time=2-00:00:00
#SBATCH --output=/lustre/fs0/scratch/jmooney/latent_profile/logs/slurm/%x-%A_%a.out
#SBATCH --container-image=/lustre/fs0/scratch/jmooney/sqsh-files/nvidia+vllm+25.09-py3.sqsh
#SBATCH --container-mounts=/home/jmooney:/home/jmooney,/lustre/fs0/scratch/jmooney:/lustre/fs0/scratch/jmooney,/lustre/fs0/scratch/shared:/lustre/fs0/scratch/shared

set -euo pipefail

PROJECT_DIR="/home/jmooney/society-sim"
PYTHON_BIN="python3"
RUN_ROOT="${RUN_ROOT:-/lustre/fs0/scratch/jmooney/latent_profile}"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH is required}"
MANIFEST_PATH="${MANIFEST_PATH:?MANIFEST_PATH is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

VENV_ROOT="${VENV_ROOT:-/lustre/fs0/scratch/jmooney/venvs}"
VENV_NAME="${VENV_NAME:-society-sim-latent-profile}"
VENV_DIR="${VENV_DIR:-$VENV_ROOT/$VENV_NAME}"

CACHE_ROOT="$RUN_ROOT/cache"
ENROOT_ROOT="/lustre/fs0/scratch/jmooney/enroot"
HF_HOME="$CACHE_ROOT/hf"
export HF_HOME
export HF_TOKEN="${HF_TOKEN:-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}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export TMPDIR="$RUN_ROOT/tmp"
export ENROOT_MAX_PROCESSORS="${ENROOT_MAX_PROCESSORS:-2}"
export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-$ENROOT_ROOT/data}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-$ENROOT_ROOT/cache}"
export ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH:-$ENROOT_ROOT/tmp}"

STATUS_ROOT="$RUN_ROOT/status"
LOG_ROOT="$RUN_ROOT/logs/array_tasks"
SLURM_LOG_ROOT="$RUN_ROOT/logs/slurm"
MANIFEST_ROOT="$RUN_ROOT/manifests"

mkdir -p \
  "$VENV_ROOT" \
  "$CACHE_ROOT" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" \
  "$XDG_CACHE_HOME" \
  "$MPLCONFIGDIR" \
  "$PIP_CACHE_DIR" \
  "$TMPDIR" \
  "$ENROOT_DATA_PATH" \
  "$ENROOT_CACHE_PATH" \
  "$ENROOT_TEMP_PATH" \
  "$STATUS_ROOT" \
  "$LOG_ROOT" \
  "$SLURM_LOG_ROOT" \
  "$MANIFEST_ROOT"

MODEL_ID="$(sed -n "$((TASK_ID + 1))p" "$MANIFEST_PATH")"
if [[ -z "${MODEL_ID:-}" ]]; then
  echo "No model id found for task id $TASK_ID in $MANIFEST_PATH" >&2
  exit 3
fi

cd "$PROJECT_DIR"
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
unset MASTER_ADDR MASTER_PORT
unset PMI_RANK PMI_SIZE
unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE OMPI_COMM_WORLD_LOCAL_RANK
unset MV2_COMM_WORLD_RANK MV2_COMM_WORLD_SIZE MV2_COMM_WORLD_LOCAL_RANK
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=-1

VENV_LOCK="${VENV_ROOT}/.${VENV_NAME}.lock"
{
  flock -x 200
  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Virtualenv missing python executable at $VENV_DIR/bin/python" >&2
    exit 4
  fi
  if ! "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import importlib
mods = ["numpy", "yaml", "requests", "openai", "transformers", "torch"]
for mod in mods:
    importlib.import_module(mod)
PY
  then
    "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel >/dev/null
    "$VENV_DIR/bin/python" -m pip install numpy pyyaml requests openai transformers torch >/dev/null
  fi
} 200>"$VENV_LOCK"

source "$VENV_DIR/bin/activate"

PORT_LOCK="$TMPDIR/port_allocator.lock"
PIPELINE_PORT="$(
  {
    flock -x 201
    python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
  } 201>"$PORT_LOCK"
)"
export PIPELINE_PORT
export VLLM_PORT="$PIPELINE_PORT"
export LATENT_PROFILE_OUTPUT_ROOT="$RUN_ROOT"

TASK_LOG="$LOG_ROOT/task_${TASK_ID}_model_${MODEL_ID}.log"
exec > >(tee -a "$TASK_LOG") 2>&1

STATUS_FILE="$STATUS_ROOT/task_${TASK_ID}_model_${MODEL_ID}.json"
echo "{\"status\":\"START\",\"task_id\":${TASK_ID},\"model_id\":\"${MODEL_ID}\",\"job_id\":\"${SLURM_JOB_ID:-unknown}\",\"array_job_id\":\"${SLURM_ARRAY_JOB_ID:-unknown}\",\"config_path\":\"${CONFIG_PATH}\",\"host\":\"$(hostname)\",\"port\":${PIPELINE_PORT}}" > "$STATUS_FILE"

echo "[$(date)] Running task ${TASK_ID} for model ${MODEL_ID}"
echo "[$(date)] Config: ${CONFIG_PATH}"
echo "[$(date)] Port: ${PIPELINE_PORT}"
echo "[$(date)] HF token present: ${HF_TOKEN:+yes}"

if python pipeline.py resume --config "$CONFIG_PATH" --model-ids "$MODEL_ID" --skip-analysis --port "$PIPELINE_PORT"; then
  echo "{\"status\":\"SUCCESS\",\"task_id\":${TASK_ID},\"model_id\":\"${MODEL_ID}\",\"job_id\":\"${SLURM_JOB_ID:-unknown}\",\"array_job_id\":\"${SLURM_ARRAY_JOB_ID:-unknown}\",\"config_path\":\"${CONFIG_PATH}\",\"host\":\"$(hostname)\",\"port\":${PIPELINE_PORT}}" > "$STATUS_FILE"
else
  EXIT_CODE=$?
  echo "{\"status\":\"FAILED\",\"task_id\":${TASK_ID},\"model_id\":\"${MODEL_ID}\",\"job_id\":\"${SLURM_JOB_ID:-unknown}\",\"array_job_id\":\"${SLURM_ARRAY_JOB_ID:-unknown}\",\"config_path\":\"${CONFIG_PATH}\",\"host\":\"$(hostname)\",\"port\":${PIPELINE_PORT},\"exit_code\":${EXIT_CODE}}" > "$STATUS_FILE"
  exit "$EXIT_CODE"
fi
