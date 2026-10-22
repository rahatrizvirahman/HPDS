#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 tests/check_predictions.py --cuda
k=${K:-3}
read -r -a datasets <<< "${DATASETS:-small medium large}"
read -r -a blocks <<< "${BLOCK_SIZES:-64 128 256}"
if ((${#datasets[@]} == 0 || ${#blocks[@]} == 0)); then exit 1; fi
mkdir -p results
scratch=$(mktemp -d results/verify.XXXXXX)
trap 'rm -rf "$scratch"' EXIT
for dataset in "${datasets[@]}"; do
    case "$dataset" in small|medium|large) ;; *) echo "Unknown dataset $dataset" >&2; exit 1;; esac
    ./serial "datasets/$dataset-train.arff" "datasets/$dataset-test.arff" "$k" "$scratch/serial.txt"
    test -s "$scratch/serial.txt"
    for block in "${blocks[@]}"; do
        ./cuda "datasets/$dataset-train.arff" "datasets/$dataset-test.arff" "$k" "$block" "$scratch/cuda.txt"
        test -s "$scratch/cuda.txt"
        if ! cmp -s "$scratch/serial.txt" "$scratch/cuda.txt"; then
            echo "Prediction mismatch for $dataset at block size $block" >&2
            diff -u "$scratch/serial.txt" "$scratch/cuda.txt" >&2 || true
            exit 1
        fi
        echo "PASS $dataset k=$k block=$block, all predictions match serial"
    done
done
echo "ALL PREDICTIONS MATCH SERIAL"
