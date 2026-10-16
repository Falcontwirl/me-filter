# Sourced by job scripts: loads config and activates the env.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
module load "$CONDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
mkdir -p "$REPO_DIR/logs" "$HF_HOME"
cd "$REPO_DIR"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo no-gpu)"
