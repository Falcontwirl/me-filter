#!/bin/bash
#SBATCH -J compositional
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -o logs/%x-%A_%a.out
# Submit from the repo root (logs/ must exist). To request a specific GPU type, add
# a --constraint line; list types with: sinfo -p gpu -o "%N %G"
source slurm/common.sh
python experiments/05_compositional.py "$@"
