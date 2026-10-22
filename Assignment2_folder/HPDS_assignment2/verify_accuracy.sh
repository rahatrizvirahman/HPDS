#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# RESULTS_ROOT (default results) and RUN_ID select where evidence is written.
results_root=${RESULTS_ROOT:-results}
run_id=${RUN_ID:+$RUN_ID.}
python3 tests/check_predictions.py --cuda
k=${K:-3}
read -r -a datasets <<< "${DATASETS:-small medium large}"
read -r -a blocks <<< "${BLOCK_SIZES:-64 128 256}"
if ((${#datasets[@]} == 0 || ${#blocks[@]} == 0)); then exit 1; fi
mkdir -p "$results_root"
scratch=$(mktemp -d "$results_root/verify.${run_id}XXXXXX")
echo "Verification artifacts $scratch"
for dataset in "${datasets[@]}"; do
    case "$dataset" in small|medium|large) ;; *) echo "Unknown dataset $dataset" >&2; exit 1;; esac
    ./serial "datasets/$dataset-train.arff" "datasets/$dataset-test.arff" "$k" "$scratch/$dataset-serial.txt"
    test -s "$scratch/$dataset-serial.txt"
    for block in "${blocks[@]}"; do
        ./cuda "datasets/$dataset-train.arff" "datasets/$dataset-test.arff" "$k" "$block" "$scratch/$dataset-cuda-b$block.txt"
        test -s "$scratch/$dataset-cuda-b$block.txt"
        if ! cmp -s "$scratch/$dataset-serial.txt" "$scratch/$dataset-cuda-b$block.txt"; then
            echo "Prediction mismatch for $dataset at block size $block" >&2
            diff -u "$scratch/$dataset-serial.txt" "$scratch/$dataset-cuda-b$block.txt" >&2 || true
            exit 1
        fi
        echo "PASS $dataset k=$k block=$block, all predictions match serial"
    done
done
echo "ALL PREDICTIONS MATCH SERIAL"
