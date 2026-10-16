#!/bin/bash
#SBATCH -J locate_build
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -o logs/%x-%A_%a.out

# Submit from the repo root (logs/ must exist). To request a specific GPU type, add
# a --constraint line; list types with: sinfo -p gpu -o "%N %G"
source slurm/common.sh
# usage: sbatch slurm/locate_build.sbatch <run> <pool> <seed> [extra args]
#   e.g. sbatch slurm/locate_build.sbatch discovery_s0 seen 0
RUN=$1; POOL=$2; SEED=$3; shift 3
python experiments/02_locate.py build --run "$RUN" --pool "$POOL" --seed "$SEED" "$@"
