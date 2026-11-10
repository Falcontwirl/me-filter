# Sourced by job scripts: loads config and activates the env.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
# Put the env first on PATH instead of `conda activate`: Oscar's anaconda3 module
# lost execute permission on its conda binary (Oct 2026), which breaks activation.
export PATH="$HOME/.conda/envs/$CONDA_ENV/bin:$PATH"
mkdir -p "$REPO_DIR/logs" "$HF_HUB_CACHE"
cd "$REPO_DIR"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo no-gpu)"
