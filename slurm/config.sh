# Edit these once for your Oscar account. Sourced by every job script.
# Check partitions and GPU types with:  sinfo -p gpu -o "%N %G %m"
#   and CCV docs: https://docs.ccv.brown.edu/oscar
export REPO_DIR="${REPO_DIR:-$HOME/me-filter}"
export SCRATCH_DIR="${SCRATCH_DIR:-$HOME/scratch/me-filter}"   # HF model cache (~16 GB)
export CONDA_ENV="${CONDA_ENV:-filtergen}"
export CONDA_MODULE="${CONDA_MODULE:-miniconda3}"              # `module avail conda` to confirm the name

export HF_HOME="$SCRATCH_DIR/hf"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/third_party/filter"
export TOKENIZERS_PARALLELISM=false
# HF token for the gated Llama weights: run `huggingface-cli login` once, or
# put `export HF_TOKEN=...` in ~/.bashrc (never in this repo).
