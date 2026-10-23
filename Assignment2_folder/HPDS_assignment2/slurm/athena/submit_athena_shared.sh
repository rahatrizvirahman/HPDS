#!/usr/bin/env bash
# Submit one Athena job for the shared-memory milestone and print the job ID.
# Usage: bash slurm/athena/submit_athena_shared.sh small | verify | benchmark
# Same verified settings as submit_athena.sh (gpu-a100, one GPU, QOS normal, cuda/12.3).
# GPU_MODEL=a100 (default) or h100. Evidence goes to results/athena-shared (a100) or results/athena-shared-h100 so the
# frozen baseline archive inputs stay untouched and different hardware is never mixed. EXCLUDE names nodes to avoid.
# Both strategies are run so the refactored global kernel is re-verified and compared in one sweep.
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${GPU_MODEL:-a100}" in
    a100) partition=gpu-a100; sm=80; suffix=athena-shared ;;
    h100) partition=gpu-h100; sm=90; suffix=athena-shared-h100 ;;
    *) echo 'GPU_MODEL must be a100 or h100' >&2; exit 1 ;;
esac
mkdir -p "slurm/logs/$suffix" "results/$suffix"
export MODULES='cuda/12.3'
export RESULTS_ROOT="results/$suffix"
export K=3 BLOCK_SIZES='64 128 256' STRATEGIES='global shared' TILE_ROWS=256
export NVCCFLAGS="-O2 -std=c++11 --fmad=false -gencode arch=compute_$sm,code=sm_$sm"
common=(--parsable --partition="$partition" --qos=normal --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=1
        --output="slurm/logs/$suffix/job-%j.log")
if [[ -n ${EXCLUDE:-} ]]; then common+=(--exclude="$EXCLUDE"); fi
case "${1:-}" in
    small)     export DATASETS=small;                  args=(--job-name=athena-shared-small --time=00:10:00 slurm/baseline_job.sh verify) ;;
    verify)    export DATASETS='small medium large';   args=(--job-name=athena-shared-verify --time=00:20:00 slurm/baseline_job.sh verify) ;;
    benchmark) export DATASETS='small medium large' REPS=3; args=(--job-name=athena-shared-bench --time=00:40:00 slurm/baseline_job.sh benchmark) ;;
    *) echo 'Usage: submit_athena_shared.sh small|verify|benchmark' >&2; exit 1 ;;
esac
printf '%s sbatch %s\n' "$(date -u +%FT%TZ)" "${common[*]} ${args[*]} GPU_MODEL=${GPU_MODEL:-a100} NVCCFLAGS=[$NVCCFLAGS] (DATASETS=$DATASETS STRATEGIES=$STRATEGIES TILE_ROWS=$TILE_ROWS)" >> "results/$suffix/submissions.txt"
sbatch "${common[@]}" "${args[@]}"
