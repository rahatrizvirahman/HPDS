#!/bin/bash
# Verifies that threaded, openmp and mpi reproduce the serial accuracy exactly
# for every dataset. The rubric requires all 9 program/dataset combinations to
# match, so this is the correctness gate to run before any experiment.
#
# Usage: ./verify_accuracy.sh [k] [worker counts...]
#   ./verify_accuracy.sh            # k=3, workers 1 2 4 8 16
#   ./verify_accuracy.sh 3 1 2 4    # k=3, workers 1 2 4

K=${1:-3}
shift 2>/dev/null
WORKERS=${@:-"1 2 4 8 16"}

DATASETS="small medium large"
PASS=0
FAIL=0

# Pulls the "Accuracy was NN.NN%" field out of a program's stdout
acc() { sed -n 's/.*Accuracy was \([0-9.]*\)%.*/\1/p'; }

echo "k = $K, worker counts: $WORKERS"
echo

for d in $DATASETS; do
    TR="datasets/$d-train.arff"
    TE="datasets/$d-test.arff"

    REF=$(./serial "$TR" "$TE" "$K" | acc)
    echo "=== $d (serial reference accuracy: $REF%) ==="

    for w in $WORKERS; do
        T=$(./threaded "$TR" "$TE" "$K" "$w"            | acc)
        O=$(./openmp   "$TR" "$TE" "$K" "$w"            | acc)
        M=$(mpirun --oversubscribe -np "$w" ./mpi "$TR" "$TE" "$K" 2>/dev/null | acc)

        for pair in "threaded:$T" "openmp:$O" "mpi:$M"; do
            name=${pair%%:*}; val=${pair#*:}
            if [ "$val" == "$REF" ]; then
                PASS=$((PASS+1))
            else
                FAIL=$((FAIL+1))
                echo "  MISMATCH  $name w=$w  got '$val' expected '$REF'"
            fi
        done
        printf "  w=%-4s threaded=%-7s openmp=%-7s mpi=%-7s\n" "$w" "$T" "$O" "$M"
    done
    echo
done

echo "======================================"
echo "  checks passed: $PASS    failed: $FAIL"
[ "$FAIL" -eq 0 ] && echo "  ALL ACCURACIES MATCH SERIAL" || echo "  ACCURACY MISMATCHES PRESENT"
echo "======================================"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
