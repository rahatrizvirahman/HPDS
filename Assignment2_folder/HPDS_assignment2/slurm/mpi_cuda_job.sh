#!/usr/bin/env bash
#SBATCH --job-name=knn-mpi-cuda
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# Pass the verified partition, node, task and GPU request to sbatch. Load Open MPI and CUDA through MODULES.
# MPI_LAUNCH, LAYOUTS, RESULTS_ROOT and the other variables are described in verify_mpi_cuda.sh and benchmark_mpi_cuda.py.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_SUBMIT_DIR:?Submit from HPDS_assignment2}"
cd "$SLURM_SUBMIT_DIR"
test -f mpi_cuda.cu || { echo 'Submit from Assignment2_folder/HPDS_assignment2' >&2; exit 1; }
if [[ -n ${MODULES:-} ]]; then
    type module >/dev/null 2>&1 || { echo 'module is unavailable' >&2; exit 1; }
    read -r -a requested_modules <<< "$MODULES"
    module load "${requested_modules[@]}"
fi
if [[ -n ${RESULTS_ROOT:-} && -z ${RUN_ID:-} ]]; then export RUN_ID="job$SLURM_JOB_ID"; fi
results_root=${RESULTS_ROOT:-results}
export RESULTS_ROOT="$results_root"
mkdir -p "$results_root"
export TMPDIR="$PWD/$results_root/tmp-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
git rev-parse HEAD
git status --short > "$results_root/job-$SLURM_JOB_ID.status.txt"
git diff -- . > "$results_root/job-$SLURM_JOB_ID.patch"
module list 2>&1
lscpu
hostname
scontrol show hostnames "$SLURM_JOB_NODELIST"
date -u
printf 'Slurm job %s\n' "$SLURM_JOB_ID"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
scontrol show job "$SLURM_JOB_ID" | tee "$results_root/job-$SLURM_JOB_ID.scontrol.txt" | egrep 'NumNodes|NumCPUs|TRES|Gres|NodeList' || true
nvcc --version
g++ --version
mpicxx --version | head -1
ompi_info | head -3
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -c 'echo "$(hostname) $(nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader | tr "\n" ";")"'
make -B serial cuda mpi_cuda
python3 tests/test_benchmark.py
python3 tests/test_benchmark_mpi.py
case "${1:-verify}" in
    verify) bash ./verify_mpi_cuda.sh ;;
    benchmark) python3 benchmark_mpi_cuda.py ;;
    *) echo 'Expected verify or benchmark' >&2; exit 1 ;;
esac
