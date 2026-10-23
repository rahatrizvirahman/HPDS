#!/usr/bin/env bash
# Submit one Athena MPI+CUDA job and print the job ID.
# Usage: bash slurm/athena/submit_athena_mpi.sh small|verify|benchmark|multinode-small|multinode-verify|multinode-benchmark
# GPU_MODEL=a100 (default), h100 or v100 selects the partition and the CUDA architecture. GPUS (default 4) sets the
# GPUs of single-node jobs, GPUS_PER_NODE (default 2) those of multinode jobs, and LAYOUTS overrides the rank layouts. Evidence for each model is kept in results/athena-mpi (a100), athena-mpi-h100
# and athena-mpi-v100 so that results from different hardware are never mixed.
# Settings: partition gpu-a100 (4 GPUs per node), QOS normal, modules cuda/12.3 and mpi/openmpi-4.1.6.
# mpi/openmpi-4.1.8 is avoided because it loads gcc-toolset 13, which CUDA 12.3 does not support.
# Ranks are launched with srun and the pmix plugin, one GPU per task. Evidence goes to results/athena-mpi.
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${GPU_MODEL:-a100}" in
    a100) partition=gpu-a100; sm=80; suffix=athena-mpi ;;
    h100) partition=gpu-h100; sm=90; suffix=athena-mpi-h100 ;;
    v100) partition=gpu-v100; sm=70; suffix=athena-mpi-v100 ;;
    *) echo 'GPU_MODEL must be a100, h100 or v100' >&2; exit 1 ;;
esac
mkdir -p "slurm/logs/$suffix" "results/$suffix"
export MODULES='cuda/12.3 mpi/openmpi-4.1.6'
export RESULTS_ROOT="results/$suffix"
export K=3 BLOCK_SIZES='64 128 256' STRATEGIES='global shared' TILE_ROWS=256
export STRATEGY=shared BLOCK_SIZE=128
export NVCCFLAGS="-O2 -std=c++11 --fmad=false -gencode arch=compute_$sm,code=sm_$sm"
export MPI_LAUNCH='srun --mpi=pmix --nodes={nodes} --ntasks-per-node={per_node} --gpus-per-task=1 --cpus-per-task=1'
common=(--parsable --partition="$partition" --qos=normal --cpus-per-task=1 --output="slurm/logs/$suffix/job-%j.log")
# EXCLUDE names nodes to avoid. athena534 failed step creation and two-node launches (see ATHENA_MPI_CUDA_RESULTS.md).
if [[ -n ${EXCLUDE:-} ]]; then common+=(--exclude="$EXCLUDE"); fi
gpus=${GPUS:-4}; per_node=${GPUS_PER_NODE:-2}
one_node=(--nodes=1 --ntasks="$gpus" --gres="gpu:$gpus")
two_nodes=(--nodes=2 --ntasks-per-node="$per_node" --gres="gpu:$per_node")
case "${1:-}" in
    small)  export DATASETS=small BLOCK_SIZES=128 LAYOUTS="${LAYOUTS:-1x1 1x2 1x3 1x4}"; export STRATEGIES='global shared'
            args=("${one_node[@]}" --job-name=athena-mpi-small --time=00:15:00 slurm/mpi_cuda_job.sh verify) ;;
    verify) export DATASETS='small medium large' LAYOUTS="${LAYOUTS:-1x1 1x2 1x3 1x4}"
            args=("${one_node[@]}" --job-name=athena-mpi-verify --time=00:30:00 slurm/mpi_cuda_job.sh verify) ;;
    benchmark) export DATASETS='small medium large' REPS=3 LAYOUTS="${LAYOUTS:-1x1 1x2 1x4}"
            args=("${one_node[@]}" --job-name=athena-mpi-bench --time=00:30:00 slurm/mpi_cuda_job.sh benchmark) ;;
    multinode-small) export DATASETS=small BLOCK_SIZES=128 LAYOUTS="${LAYOUTS:-1x1 1x2 2x1 2x2}"
            args=("${two_nodes[@]}" --job-name=athena-mpi-mn-small --time=00:15:00 slurm/mpi_cuda_job.sh verify) ;;
    multinode-verify) export DATASETS='small medium large' LAYOUTS="${LAYOUTS:-1x1 1x2 2x1 2x2}"
            args=("${two_nodes[@]}" --job-name=athena-mpi-mn-verify --time=00:30:00 slurm/mpi_cuda_job.sh verify) ;;
    multinode-benchmark) export DATASETS='small medium large' REPS=3 LAYOUTS="${LAYOUTS:-1x1 1x2 2x1 2x2}"
            args=("${two_nodes[@]}" --job-name=athena-mpi-mn-bench --time=00:30:00 slurm/mpi_cuda_job.sh benchmark) ;;
    *) echo 'Usage: submit_athena_mpi.sh small|verify|benchmark|multinode-small|multinode-verify|multinode-benchmark' >&2; exit 1 ;;
esac
printf '%s sbatch %s (MODULES=%s DATASETS=%s LAYOUTS=%s STRATEGIES=%s STRATEGY=%s BLOCK_SIZE=%s TILE_ROWS=%s MPI_LAUNCH=%s)\n' \
    "$(date -u +%FT%TZ)" "${common[*]} ${args[*]} GPU_MODEL=${GPU_MODEL:-a100} NVCCFLAGS=[$NVCCFLAGS]" "$MODULES" "$DATASETS" "$LAYOUTS" "$STRATEGIES" "$STRATEGY" "$BLOCK_SIZE" "$TILE_ROWS" "$MPI_LAUNCH" >> "results/$suffix/submissions.txt"
sbatch "${common[@]}" "${args[@]}"
