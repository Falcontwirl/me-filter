#!/bin/bash
#SBATCH -J locate_train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -o logs/%x-%A_%a.out
#SBATCH --array=0-3
# Submit from the repo root (logs/ must exist). To request a specific GPU type, add
# a --constraint line; list types with: sinfo -p gpu -o "%N %G"
source slurm/common.sh
# usage: sbatch slurm/locate_train.sbatch <run>                       (lambda sweep, one task per lambda)
#        sbatch --array=0 slurm/locate_train.sbatch <run> <lambda>    (one chosen lambda)
LAMBDAS=(0.005 0.01 0.02 0.05)
RUN=$1
LAM=${2:-${LAMBDAS[$SLURM_ARRAY_TASK_ID]}}
python experiments/02_locate.py train --run "$RUN" --lam "$LAM"
