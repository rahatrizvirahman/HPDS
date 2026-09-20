#!/bin/bash
# Runs the full experiment sweep on a single shared-memory machine (maple).
# Every configuration is repeated REPS times; collect_results.py averages them.
#
# Usage:
#   ./run_experiments.sh                     # everything, 3 reps
#   REPS=1 DATASETS="small" ./run_experiments.sh
#   WORKER_COUNTS="1 2 4" ./run_experiments.sh
#
# Output: results/maple.csv

set -u
cd "$(dirname "$0")"
source ./bench_common.sh

CSV=results/maple.csv
mkdir -p results
# Only write the header if the file doesn't exist yet, so scoping DATASETS to
# resume a partial sweep (e.g. DATASETS="medium large") appends to prior
# results instead of wiping them out.
[ -f "$CSV" ] || csv_header > "$CSV"

echo "Platform: maple  |  k=$K  reps=$REPS"
echo "Datasets: $DATASETS"
echo "Workers:  $WORKER_COUNTS"
echo "Logical CPUs visible to this shell: $(nproc)"
echo

# Serial baseline. Runs once per dataset per rep, with no worker dimension.
for d in $DATASETS; do
    for r in $(seq 1 "$REPS"); do
        line=$(./serial datasets/$d-train.arff datasets/$d-test.arff $K)
        emit_row "$CSV" maple serial "$d" 1 1 "$r" "$line"
    done
done

for d in $DATASETS; do
    for w in $WORKER_COUNTS; do
        for r in $(seq 1 "$REPS"); do
            line=$(./threaded datasets/$d-train.arff datasets/$d-test.arff $K $w)
            emit_row "$CSV" maple threaded "$d" "$w" 1 "$r" "$line"

            line=$(./openmp datasets/$d-train.arff datasets/$d-test.arff $K $w)
            emit_row "$CSV" maple openmp "$d" "$w" 1 "$r" "$line"

            line=$(mpirun --oversubscribe -np $w ./mpi datasets/$d-train.arff datasets/$d-test.arff $K 2>/dev/null)
            emit_row "$CSV" maple mpi "$d" "$w" 1 "$r" "$line"
        done
    done
done

echo
echo "Wrote $CSV ($(($(wc -l < "$CSV") - 1)) rows)"
