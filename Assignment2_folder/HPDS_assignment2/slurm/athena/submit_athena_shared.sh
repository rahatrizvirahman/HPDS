#!/usr/bin/env bash
# Submit one Athena job for the shared-memory milestone and print the job ID.
# Usage: bash slurm/athena/submit_athena_shared.sh small | verify | benchmark
# Same verified settings as submit_athena.sh (gpu-a100, one GPU, QOS normal, cuda/12.3).
# Evidence goes to results/athena-shared so the frozen baseline archive inputs stay untouched.
# Both strategies are run so the refactored global kernel is re-verified and compared in one sweep.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p slurm/logs/athena-shared results/athena-shared
export MODULES='cuda/12.3'
export RESULTS_ROOT=results/athena-shared
export K=3 BLOCK_SIZES='64 128 256' STRATEGIES='global shared' TILE_ROWS=256
export NVCCFLAGS='-O2 -std=c++11 --fmad=false -gencode arch=compute_80,code=sm_80'
common=(--parsable --partition=gpu-a100 --qos=normal --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=1
        --output=slurm/logs/athena-shared/job-%j.log)
case "${1:-}" in
    small)     export DATASETS=small;                  args=(--job-name=athena-shared-small --time=00:10:00 slurm/baseline_job.sh verify) ;;
    verify)    export DATASETS='small medium large';   args=(--job-name=athena-shared-verify --time=00:20:00 slurm/baseline_job.sh verify) ;;
    benchmark) export DATASETS='small medium large' REPS=3; args=(--job-name=athena-shared-bench --time=00:40:00 slurm/baseline_job.sh benchmark) ;;
    *) echo 'Usage: submit_athena_shared.sh small|verify|benchmark' >&2; exit 1 ;;
esac
printf '%s sbatch %s\n' "$(date -u +%FT%TZ)" "${common[*]} ${args[*]} (DATASETS=$DATASETS STRATEGIES=$STRATEGIES TILE_ROWS=$TILE_ROWS)" >> results/athena-shared/submissions.txt
sbatch "${common[@]}" "${args[@]}"
