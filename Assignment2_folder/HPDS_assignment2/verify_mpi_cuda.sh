#!/usr/bin/env bash
# Verify MPI+CUDA predictions against serial. Run inside a Slurm allocation with the needed GPUs.
# LAYOUTS lists NODESxRANKS_PER_NODE values. MPI_LAUNCH is the launcher template.
set -euo pipefail
cd "$(dirname "$0")"
results_root=${RESULTS_ROOT:-results}
run_id=${RUN_ID:+$RUN_ID.}
k=${K:-3}
read -r -a datasets <<< "${DATASETS:-small medium large}"
read -r -a layouts <<< "${LAYOUTS:-1x1 1x2 1x3 1x4}"
read -r -a strategies <<< "${STRATEGIES:-global shared}"
read -r -a blocks <<< "${BLOCK_SIZES:-64 128 256}"
tile=${TILE_ROWS:-256}
# A default containing braces must not sit inside ${...:-...}, which would end at the first brace.
default_launch='mpirun -n {ranks}'
launch_template=${MPI_LAUNCH:-$default_launch}
python3 tests/check_mpi_predictions.py
mkdir -p "$results_root"
scratch=$(mktemp -d "$results_root/mpi-verify.${run_id}XXXXXX")
echo "MPI verification artifacts $scratch"
for dataset in "${datasets[@]}"; do
    case "$dataset" in small|medium|large) ;; *) echo "Unknown dataset $dataset" >&2; exit 1;; esac
    ./serial "datasets/$dataset-train.arff" "datasets/$dataset-test.arff" "$k" "$scratch/$dataset-serial.txt"
    for layout in "${layouts[@]}"; do
        nodes=${layout%x*}; per_node=${layout#*x}
        # Quoted patterns keep the braces literal.
        pattern_nodes='{nodes}'; pattern_per_node='{per_node}'; pattern_ranks='{ranks}'
        launch=${launch_template//"$pattern_nodes"/$nodes}
        launch=${launch//"$pattern_per_node"/$per_node}
        launch=${launch//"$pattern_ranks"/$((nodes * per_node))}
        for strategy in "${strategies[@]}"; do
            extra=(); if [[ $strategy == shared ]]; then extra=("--strategy=shared" "--tile=$tile"); fi
            for block in "${blocks[@]}"; do
                out="$scratch/$dataset-$strategy-L$layout-b$block.txt"
                # shellcheck disable=SC2086
                $launch ./mpi_cuda "datasets/$dataset-train.arff" "datasets/$dataset-test.arff" "$k" "$block" "$out" ${extra[@]+"${extra[@]}"} \
                    > "$scratch/$dataset-$strategy-L$layout-b$block.log" 2>&1 || { cat "$scratch/$dataset-$strategy-L$layout-b$block.log" >&2; exit 1; }
                if ! cmp -s "$scratch/$dataset-serial.txt" "$out"; then
                    echo "Prediction mismatch for $dataset $strategy layout $layout block $block" >&2
                    diff -u "$scratch/$dataset-serial.txt" "$out" >&2 || true
                    exit 1
                fi
                echo "PASS $dataset $strategy layout=$layout k=$k block=$block, all predictions match serial"
            done
        done
    done
done
echo "ALL MPI PREDICTIONS MATCH SERIAL"
