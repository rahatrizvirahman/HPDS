#!/usr/bin/env bash
# Verify pthreads, OpenMP and MPI CPU predictions against serial. Run inside a Slurm allocation.
# WORKERS lists worker counts. MPI_LAUNCH is a template such as "srun --mpi=pmix --ntasks={ranks}".
set -euo pipefail
cd "$(dirname "$0")"
results_root=${RESULTS_ROOT:-results}
run_id=${RUN_ID:+$RUN_ID.}
k=${K:-3}
read -r -a datasets <<< "${DATASETS:-small medium large}"
read -r -a workers <<< "${WORKERS:-1 2 4 8 16}"
default_launch='mpirun -n {ranks}'
launch_template=${MPI_LAUNCH-$default_launch}
python3 tests/check_cpu_predictions.py
mkdir -p "$results_root"
scratch=$(mktemp -d "$results_root/cpu-verify.${run_id}XXXXXX")
echo "CPU verification artifacts $scratch"
pattern_ranks='{ranks}'
for dataset in "${datasets[@]}"; do
    ./serial "datasets/$dataset-train.arff" "datasets/$dataset-test.arff" "$k" "$scratch/$dataset-serial.txt"
    for w in "${workers[@]}"; do
        for program in threaded openmp mpi; do
            out="$scratch/$dataset-$program-w$w.txt"
            if [[ $program == mpi ]]; then
                [[ -n $launch_template ]] || continue
                launch=${launch_template//"$pattern_ranks"/$w}
                # shellcheck disable=SC2086
                $launch ./mpi "datasets/$dataset-train.arff" "datasets/$dataset-test.arff" "$k" "$out" > "$scratch/$dataset-$program-w$w.log" 2>&1 || { cat "$scratch/$dataset-$program-w$w.log" >&2; exit 1; }
            else
                "./$program" "datasets/$dataset-train.arff" "datasets/$dataset-test.arff" "$k" "$w" "$out" > "$scratch/$dataset-$program-w$w.log" 2>&1 || { cat "$scratch/$dataset-$program-w$w.log" >&2; exit 1; }
            fi
            if ! cmp -s "$scratch/$dataset-serial.txt" "$out"; then
                echo "Prediction mismatch for $dataset $program workers $w" >&2
                diff -u "$scratch/$dataset-serial.txt" "$out" >&2 || true
                exit 1
            fi
            echo "PASS $dataset $program workers=$w k=$k, all predictions match serial"
        done
    done
done
echo "ALL CPU PREDICTIONS MATCH SERIAL"
