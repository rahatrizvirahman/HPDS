#!/usr/bin/env bash
# Submit one Athena CPU comparison job and print the job ID.
# Usage: bash slurm/athena/submit_athena_cpu.sh small|verify|benchmark
# CPU partition `cpu`, QOS normal, one node with 16 cores for up to 16 threads or 16 MPI ranks, modules mpi/openmpi-4.1.6.
# Ranks are launched with srun --mpi=pmix, which worked within a single node. Evidence goes to results/athena-cpu.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p slurm/logs/athena-cpu results/athena-cpu
export MODULES='mpi/openmpi-4.1.6'
export RESULTS_ROOT=results/athena-cpu
export K=3 WORKERS='1 2 4 8 16'
export MPI_LAUNCH='srun --mpi=pmix --ntasks={ranks} --cpus-per-task=1'
common=(--parsable --partition=cpu --qos=normal --nodes=1 --ntasks=16 --cpus-per-task=1 --output=slurm/logs/athena-cpu/job-%j.log)
if [[ -n ${EXCLUDE:-} ]]; then common+=(--exclude="$EXCLUDE"); fi
case "${1:-}" in
    small)     export DATASETS=small;                 args=(--job-name=athena-cpu-small --time=00:10:00 slurm/cpu_job.sh verify) ;;
    verify)    export DATASETS='small medium large';  args=(--job-name=athena-cpu-verify --time=00:20:00 slurm/cpu_job.sh verify) ;;
    benchmark) export DATASETS='small medium large' REPS=3; args=(--job-name=athena-cpu-bench --time=00:40:00 slurm/cpu_job.sh benchmark) ;;
    *) echo 'Usage: submit_athena_cpu.sh small|verify|benchmark' >&2; exit 1 ;;
esac
printf '%s sbatch %s (MODULES=%s DATASETS=%s WORKERS=%s MPI_LAUNCH=%s)\n' "$(date -u +%FT%TZ)" "${common[*]} ${args[*]}" \
    "$MODULES" "$DATASETS" "$WORKERS" "$MPI_LAUNCH" >> results/athena-cpu/submissions.txt
sbatch "${common[@]}" "${args[@]}"
