#!/bin/bash
# hickory.cs.vcu.edu variant of submit_all.sh. Run this once from
# HPDS_assignment1/ on hickory, after `make` has produced
# serial/threaded/openmp/mpi in this folder.
#
#   ./slurm/submit_all_hickory.sh
#   REPS=1 DATASETS="small" ./slurm/submit_all_hickory.sh          # quick smoke test
#   WORKER_COUNTS="1 2 4" ./slurm/submit_all_hickory.sh
#
# Differences from submit_all.sh, based on hickory's actual Slurm config
# (checked 2026-09-20):
#
#   - `sacctmgr show assoc user=$USER` shows this account capped at
#     MaxSubmitJobs=2 and MaxJobs=2, i.e. at most 2 jobs may sit in the queue
#     (pending or running) at once. submit_all.sh's "fire everything, then
#     let Slurm queue it" approach would have every sbatch past the first two
#     rejected outright (QOSMaxSubmitJobPerUserLimit), so this script submits
#     one job at a time and polls `squeue -u $USER` (via wait_for_slot below),
#     submitting the next job only once fewer than 2 of this user's jobs are
#     still queued.
#   - `sinfo -N -o "%N %P %t %c %C %m %G"` / `sinfo -s` show hickory's "gpu"
#     partition has exactly one node (hickory itself), so there is no second
#     node to place a multi-node MPI run on. The --nodes=2 sweep is dropped
#     entirely here rather than skipped only at worker count 1.
#   - `sacctmgr show qos` shows QOS "short" (requested via --qos=short in
#     knn_job_hickory.sh) caps a job at MaxCPUs=6, so WORKER_COUNTS defaults
#     to values that fit under that ceiling instead of Athena's 1..128 sweep.
#   - Every --output log lands with a "_hickory" suffix (e.g.
#     threaded_small_4_rep1_hickory.log) instead of Athena's plain
#     threaded_small_4_rep1.log, so results from both clusters can sit in the
#     same slurm/logs/ directory without colliding, and so
#     slurm/collect_hickory_results.py can tell hickory's own logs apart from
#     any Athena logs that get copied into the same folder.

set -u
cd "$(dirname "$0")/.."

K=${K:-3}
REPS=${REPS:-3}
DATASETS=${DATASETS:-"small medium large"}
WORKER_COUNTS=${WORKER_COUNTS:-"1 2 4 6"}

mkdir -p slurm/logs

# Blocks until this user has fewer than 2 jobs pending/running in Slurm,
# since the hickory account association caps MaxSubmitJobs/MaxJobs at 2.
wait_for_slot() {
    while true; do
        n=$(squeue -u "$USER" -h -t pending,running | wc -l)
        if [ "$n" -lt 2 ]; then
            return
        fi
        sleep 15
    done
}

echo "k=$K  reps=$REPS  datasets=[$DATASETS]  workers=[$WORKER_COUNTS]"
n=0

for d in $DATASETS; do
    for r in $(seq 1 "$REPS"); do
        wait_for_slot
        sbatch --job-name=serial_${d} \
               --output=slurm/logs/serial_${d}_rep${r}_hickory.log \
               slurm/knn_job_hickory.sh serial "$d" "$K" > /dev/null
        n=$((n+1))
    done
done

for d in $DATASETS; do
    for w in $WORKER_COUNTS; do
        for r in $(seq 1 "$REPS"); do

            wait_for_slot
            sbatch --job-name=thr_${d}_${w} --cpus-per-task="$w" \
                   --output=slurm/logs/threaded_${d}_${w}_rep${r}_hickory.log \
                   slurm/knn_job_hickory.sh threaded "$d" "$K" > /dev/null
            n=$((n+1))

            wait_for_slot
            sbatch --job-name=omp_${d}_${w} --cpus-per-task="$w" \
                   --output=slurm/logs/openmp_${d}_${w}_rep${r}_hickory.log \
                   slurm/knn_job_hickory.sh openmp "$d" "$K" > /dev/null
            n=$((n+1))

            wait_for_slot
            sbatch --job-name=mpi1_${d}_${w} --ntasks="$w" --nodes=1 \
                   --output=slurm/logs/mpi_1node_${d}_${w}_rep${r}_hickory.log \
                   slurm/knn_job_hickory.sh mpi "$d" "$K" > /dev/null
            n=$((n+1))

        done
    done
done

echo "Submitted $n jobs. Track with: squeue -u \$USER"
echo "Logs land in slurm/logs/ as each job finishes."
