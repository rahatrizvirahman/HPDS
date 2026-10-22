#!/usr/bin/env bash
#SBATCH --job-name=knn-cuda-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

# Pass the verified partition, GPU request, account and QOS to sbatch.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_SUBMIT_DIR:?Submit from HPDS_assignment2}"
cd "$SLURM_SUBMIT_DIR"
test -f cuda.cu || { echo 'Submit from Assignment2_folder/HPDS_assignment2' >&2; exit 1; }
# Load the cluster's compiler and CUDA modules before sbatch, or set MODULES.
if [[ -n ${MODULES:-} ]]; then
    type module >/dev/null 2>&1 || { echo 'module is unavailable' >&2; exit 1; }
    read -r -a requested_modules <<< "$MODULES"
    module load "${requested_modules[@]}"
fi
hostname
date -u
printf 'Slurm job %s\n' "$SLURM_JOB_ID"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
nvcc --version
g++ --version
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
# Force rebuilding so the log captures the actual flags and no stale binary is used.
make -B all
case "${1:-verify}" in
    verify) srun --ntasks=1 bash ./verify_accuracy.sh ;;
    benchmark) srun --ntasks=1 bash ./run_experiments.sh ;;
    *) echo 'Expected verify or benchmark' >&2; exit 1 ;;
esac
