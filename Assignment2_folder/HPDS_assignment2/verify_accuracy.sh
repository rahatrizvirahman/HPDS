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
read -r -a strategies <<< "${STRATEGIES:-global}"
tile=${TILE_ROWS:-256}
if ((${#datasets[@]} == 0 || ${#blocks[@]} == 0)); then exit 1; fi
mkdir -p "$results_root"
scratch=$(mktemp -d "$results_root/verify.${run_id}XXXXXX")
echo "Verification artifacts $scratch"
for dataset in "${datasets[@]}"; do
    case "$dataset" in small|medium|large) ;; *) echo "Unknown dataset $dataset" >&2; exit 1;; esac
    ./serial "datasets/$dataset-train.arff" "datasets/$dataset-test.arff" "$k" "$scratch/$dataset-serial.txt"
    test -s "$scratch/$dataset-serial.txt"
    for strategy in "${strategies[@]}"; do
        case "$strategy" in
            global) extra=() ;;
            shared) extra=("--strategy=shared" "--tile=$tile") ;;
            *) echo "Unknown strategy $strategy" >&2; exit 1 ;;
        esac
        # Global keeps its original file names so existing evidence layouts are unchanged.
        label=cuda; if [[ $strategy != global ]]; then label=$strategy; fi
        for block in "${blocks[@]}"; do
            out="$scratch/$dataset-$label-b$block.txt"
            ./cuda "datasets/$dataset-train.arff" "datasets/$dataset-test.arff" "$k" "$block" "$out" ${extra[@]+"${extra[@]}"}
            test -s "$out"
            if ! cmp -s "$scratch/$dataset-serial.txt" "$out"; then
                echo "Prediction mismatch for $dataset $strategy at block size $block" >&2
                diff -u "$scratch/$dataset-serial.txt" "$out" >&2 || true
                exit 1
            fi
            echo "PASS $dataset $strategy k=$k block=$block, all predictions match serial"
        done
    done
done
echo "ALL PREDICTIONS MATCH SERIAL"
