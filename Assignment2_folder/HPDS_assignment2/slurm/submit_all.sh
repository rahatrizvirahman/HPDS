#!/bin/bash
# Submits the full Athena sweep as one command: every dataset, every worker
# count, the MPI single-node vs multi-node comparison, and 3 repetitions of
# each, all via slurm/knn_job.sh. Run this once from HPDS_assignment1/ on
# Athena, after `make` has produced serial/threaded/openmp/mpi in this folder.
#
#   ./slurm/submit_all.sh
#   REPS=1 DATASETS="small" ./slurm/submit_all.sh          # quick smoke test
#   WORKER_COUNTS="1 2 4" ./slurm/submit_all.sh
#
# This submits jobs and returns immediately; it does not wait for them to
# finish. Each dataset/worker-count/rep combination needs its own Slurm
# reservation sized just for it (Slurm cannot resize a job's CPU/node count
# after it starts), so "run everything" here means "submit everything," not
# "run everything inside one job." With the defaults below this submits
# around 288 separate jobs, so check `squeue -u $USER` and let them drain
# before running slurm/knn_job.sh again or moving on to analysis; Slurm
# queues jobs beyond what's immediately available, it does not reject them.
#
# --nodes 2 for the multi-node MPI comparison is skipped at worker count 1,
# since there is only one rank to place and the single-node vs multi-node
# distinction is meaningless there.

set -u
cd "$(dirname "$0")/.."

K=${K:-3}
REPS=${REPS:-3}
DATASETS=${DATASETS:-"small medium large"}
WORKER_COUNTS=${WORKER_COUNTS:-"1 2 4 8 16 32 64 128"}

mkdir -p slurm/logs

echo "k=$K  reps=$REPS  datasets=[$DATASETS]  workers=[$WORKER_COUNTS]"
n=0

for d in $DATASETS; do
    for r in $(seq 1 "$REPS"); do
        sbatch --job-name=serial_${d} \
               --output=slurm/logs/serial_${d}_rep${r}.log \
               slurm/knn_job.sh serial "$d" "$K" > /dev/null
        n=$((n+1))
    done
done

for d in $DATASETS; do
    for w in $WORKER_COUNTS; do
        for r in $(seq 1 "$REPS"); do

            sbatch --job-name=thr_${d}_${w} --cpus-per-task="$w" \
                   --output=slurm/logs/threaded_${d}_${w}_rep${r}.log \
                   slurm/knn_job.sh threaded "$d" "$K" > /dev/null
            n=$((n+1))

            sbatch --job-name=omp_${d}_${w} --cpus-per-task="$w" \
                   --output=slurm/logs/openmp_${d}_${w}_rep${r}.log \
                   slurm/knn_job.sh openmp "$d" "$K" > /dev/null
            n=$((n+1))

            sbatch --job-name=mpi1_${d}_${w} --ntasks="$w" --nodes=1 \
                   --output=slurm/logs/mpi_1node_${d}_${w}_rep${r}.log \
                   slurm/knn_job.sh mpi "$d" "$K" > /dev/null
            n=$((n+1))

            if [ "$w" -gt 1 ]; then
                sbatch --job-name=mpiN_${d}_${w} --ntasks="$w" --nodes=2 \
                       --output=slurm/logs/mpi_multinode_${d}_${w}_rep${r}.log \
                       slurm/knn_job.sh mpi "$d" "$K" > /dev/null
                n=$((n+1))
            fi

        done
    done
done

echo "Submitted $n jobs. Track with: squeue -u \$USER"
echo "Logs land in slurm/logs/ as each job finishes."
