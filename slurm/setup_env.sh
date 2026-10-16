#!/bin/bash
# One-time environment setup. Run on a login node:  bash slurm/setup_env.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"
module load "$CONDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
bash "$REPO_DIR/third_party/fetch_upstream.sh"
conda create -y -n "$CONDA_ENV" python=3.11
conda activate "$CONDA_ENV"
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r "$REPO_DIR/requirements.txt"
mkdir -p "$SCRATCH_DIR/hf" "$REPO_DIR/logs"
python -c "import torch, transformers, nnsight, baukit; print('torch', torch.__version__, '| transformers', transformers.__version__)"
echo "Next: huggingface-cli login (needs access to meta-llama/Llama-3.1-8B-Instruct), then sbatch slurm/smoke.sbatch"
