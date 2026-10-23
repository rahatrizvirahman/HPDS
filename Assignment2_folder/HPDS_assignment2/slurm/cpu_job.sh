#!/usr/bin/env bash
#SBATCH --job-name=knn-cpu
#SBATCH --time=00:30:00
#SBATCH --mem=8G

# CPU comparison job. It copies the sources into a private work directory under RESULTS_ROOT and builds there,
# so a concurrent GPU job that rebuilds the shared checkout cannot replace these binaries mid-run.
# Pass the partition and task count to sbatch. Usage: sbatch ... slurm/cpu_job.sh verify|benchmark
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_SUBMIT_DIR:?Submit from HPDS_assignment2}"
cd "$SLURM_SUBMIT_DIR"
test -f serial.cpp || { echo 'Submit from Assignment2_folder/HPDS_assignment2' >&2; exit 1; }
if [[ -n ${MODULES:-} ]]; then
    type module >/dev/null 2>&1 || { echo 'module is unavailable' >&2; exit 1; }
    read -r -a requested_modules <<< "$MODULES"
    module load "${requested_modules[@]}"
fi
: "${RESULTS_ROOT:?Set RESULTS_ROOT, for example results/athena-cpu}"
if [[ -z ${RUN_ID:-} ]]; then export RUN_ID="job$SLURM_JOB_ID"; fi
origin=$PWD
mkdir -p "$RESULTS_ROOT"
results_abs=$(cd "$RESULTS_ROOT" && pwd)
git rev-parse HEAD
git status --short > "$results_abs/job-$SLURM_JOB_ID.status.txt"
git diff -- . > "$results_abs/job-$SLURM_JOB_ID.patch"
work="$results_abs/work-$SLURM_JOB_ID"
mkdir -p "$work"
tar --exclude=./results --exclude=./slurm/logs --exclude=./cuda --exclude=./mpi_cuda --exclude=./serial \
    --exclude=./threaded --exclude=./openmp --exclude=./mpi --exclude=__pycache__ -cf - . | tar -xf - -C "$work"
cd "$work"
export RESULTS_ROOT="$results_abs"
export TMPDIR="$work/tmp"
mkdir -p "$TMPDIR"
module list 2>&1
lscpu
hostname
date -u
printf 'Slurm job %s on %s with %s CPUs\n' "$SLURM_JOB_ID" "$(hostname)" "${SLURM_CPUS_ON_NODE:-unknown}"
scontrol show job "$SLURM_JOB_ID" | tee "$results_abs/job-$SLURM_JOB_ID.scontrol.txt" | egrep 'NumNodes|NumCPUs|TRES|NodeList' || true
g++ --version
mpicxx --version | head -1
make -B serial threaded openmp mpi
python3 tests/test_benchmark.py
python3 tests/test_benchmark_cpu.py
case "${1:-verify}" in
    verify) bash ./verify_cpu.sh ;;
    benchmark) python3 benchmark_cpu.py ;;
    *) echo 'Expected verify or benchmark' >&2; exit 1 ;;
esac
