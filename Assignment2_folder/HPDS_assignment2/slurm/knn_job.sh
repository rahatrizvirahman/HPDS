#!/bin/bash
# One Slurm script for all four programs. Which program runs is picked by
# the first argument; the sbatch command line is what sets thread/process
# counts and node placement (see the examples in README.md).
#
#   sbatch --cpus-per-task 8 --output out.log slurm/knn_job.sh threaded small 3
#   sbatch --cpus-per-task 8 --output out.log slurm/knn_job.sh openmp   small 3
#   sbatch --ntasks 8 --nodes 1 --output out.log slurm/knn_job.sh mpi small 3
#   sbatch --ntasks 8 --nodes 2 --output out.log slurm/knn_job.sh mpi small 3
# Never combine --ntasks with --ntasks-per-node in the same command; use one
# or the other. --nodes with --ntasks (as above) is fine.
#   sbatch --output out.log slurm/knn_job.sh serial small 3
#
# args: $1 = program (serial|threaded|openmp|mpi), $2 = dataset (small|medium|large), $3 = k
#SBATCH --partition=cpu-large
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
        module load mpi/openmpi-4.1.6
        mpirun -np "$SLURM_NTASKS" ./mpi "$TRAIN" "$TEST" "$K"
        ;;
    *)
        echo "Unknown program '$PROGRAM'. Use serial, threaded, openmp, or mpi." >&2
        exit 1
        ;;
esac
