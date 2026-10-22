#!/usr/bin/env bash
# Submit one Athena baseline job from HPDS_assignment2 and print the job ID.
# Usage: bash slurm/athena/submit_athena.sh small | verify | benchmark
# Verified Athena settings: partition gpu-a100, one GPU, QOS normal, module cuda/12.3.
# No account option is needed. The default association (account rahmanr12) is accepted.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p slurm/logs/athena results/athena
export MODULES='cuda/12.3'
export RESULTS_ROOT=results/athena
export K=3 BLOCK_SIZES='64 128 256'
# Explicit sm_80 code for the A100 avoids first-launch PTX JIT inside the timed regions.
export NVCCFLAGS='-O2 -std=c++11 --fmad=false -gencode arch=compute_80,code=sm_80'
common=(--parsable --partition=gpu-a100 --qos=normal --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=1
        --output=slurm/logs/athena/baseline-%j.log)
case "${1:-}" in
    small)     export DATASETS=small;                  args=(--job-name=athena-verify-small --time=00:10:00 slurm/baseline_job.sh verify) ;;
    verify)    export DATASETS='small medium large';   args=(--job-name=athena-verify-all --time=00:20:00 slurm/baseline_job.sh verify) ;;
    benchmark) export DATASETS='small medium large' REPS=3; args=(--job-name=athena-benchmark --time=00:30:00 slurm/baseline_job.sh benchmark) ;;
    *) echo 'Usage: submit_athena.sh small|verify|benchmark' >&2; exit 1 ;;
esac
printf '%s sbatch %s\n' "$(date -u +%FT%TZ)" "${common[*]} ${args[*]} (DATASETS=$DATASETS)" >> results/athena/submissions.txt
sbatch "${common[@]}" "${args[@]}"
