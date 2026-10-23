#!/usr/bin/env bash
#SBATCH --job-name=athena-discover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
# Read-only discovery on a GPU node. Runs no KNN computation.
set -uo pipefail
echo "job=${SLURM_JOB_ID:-} partition=${SLURM_JOB_PARTITION:-} node=$(hostname)"
date -u
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "MODULEPATH=${MODULEPATH:-}"
nvidia-smi 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv 2>&1
module avail cuda 2>&1
module spider cuda 2>&1 | head -30
module avail gcc 2>&1 | head -20
ls -d /usr/local/cuda* /opt/cuda* /usr/lib/cuda* 2>&1
command -v nvcc || echo "nvcc not on PATH"
find / -maxdepth 4 -name nvcc -type f 2>/dev/null | head
lscpu | egrep 'Model name|Socket|Core|^CPU\(s\)'
