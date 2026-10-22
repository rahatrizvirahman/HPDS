#!/bin/bash
# hickory.cs.vcu.edu variant of knn_job.sh. Same four programs, same
# argument order; only the Slurm directives and the MPI module line changed,
# based on hickory's actual Slurm config (checked 2026-09-20):
#
#   - `sinfo -N -o "%N %P %t %c %C %m %G"` shows a single node "hickory" in a
#     single partition "gpu" (there is no "cpu-large" partition on hickory).
#   - `sacctmgr show assoc user=$USER` shows QOS=long,short with the account's
#     default QOS "long" capped at MaxCPUs=3; `sacctmgr show qos` shows QOS
#     "short" caps at MaxCPUs=6 instead, so --qos=short is requested explicitly
#     to allow worker counts up to 6 (see WORKER_COUNTS in submit_all_hickory.sh).
#   - `module avail mpi` lists "openmpi/4.1" (not "mpi/openmpi-4.1.6" as on
#     Athena) as the available OpenMPI module.
#   - hickory's only partition is a GPU partition, and its Lua job_submit
#     plugin rejects any job with no GRES request ("You must request gpu GRES
#     with --gres=gpu:<type>:<count>", seen from a live `sbatch` attempt with
#     no --gres). This benchmark is CPU-only and never touches a GPU, but the
#     smallest available GPU type (`gpu:40g`) is requested at count 1 purely
#     to satisfy that policy, taking a minimal slice of the shared GPU pool.
#
#   sbatch --cpus-per-task 8 --output out.log slurm/knn_job_hickory.sh threaded small 3
#   sbatch --cpus-per-task 8 --output out.log slurm/knn_job_hickory.sh openmp   small 3
#   sbatch --ntasks 8 --nodes 1 --output out.log slurm/knn_job_hickory.sh mpi small 3
# Never combine --ntasks with --ntasks-per-node in the same command; use one
# or the other. --nodes with --ntasks (as above) is fine.
#   sbatch --output out.log slurm/knn_job_hickory.sh serial small 3
#
# args: $1 = program (serial|threaded|openmp|mpi), $2 = dataset (small|medium|large), $3 = k
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --gres=gpu:40g:1
#SBATCH --mem 4G
#SBATCH --time 00:20:00

PROGRAM=$1
DATASET=$2
K=$3

cd "$SLURM_SUBMIT_DIR"
TRAIN=datasets/${DATASET}-train.arff
TEST=datasets/${DATASET}-test.arff

echo "nodes used: $SLURM_JOB_NUM_NODES"

case "$PROGRAM" in
    serial)
        ./serial "$TRAIN" "$TEST" "$K"
        ;;
    threaded)
        ./threaded "$TRAIN" "$TEST" "$K" "$SLURM_CPUS_PER_TASK"
        ;;
    openmp)
        export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
        ./openmp "$TRAIN" "$TEST" "$K" "$SLURM_CPUS_PER_TASK"
        ;;
    mpi)
        module load openmpi/4.1
        mpirun -np "$SLURM_NTASKS" ./mpi "$TRAIN" "$TEST" "$K"
        ;;
    *)
        echo "Unknown program '$PROGRAM'. Use serial, threaded, openmp, or mpi." >&2
        exit 1
        ;;
esac
