#!/bin/bash
# Shared helpers for the experiment scripts. Sourced, not executed directly.

K=${K:-3}
REPS=${REPS:-3}
DATASETS=${DATASETS:-"small medium large"}
WORKER_COUNTS=${WORKER_COUNTS:-"1 2 4 8 16 32 64 128"}

# Pull the runtime in ms and the accuracy out of a program's output line
parse_ms()  { sed -n 's/.*required \([0-9]*\) ms.*/\1/p'; }
parse_acc() { sed -n 's/.*Accuracy was \([0-9.]*\)%.*/\1/p'; }

csv_header() { echo "platform,version,dataset,workers,nodes,rep,time_ms,accuracy"; }

# emit_row <csv file> <platform> <version> <dataset> <workers> <nodes> <rep> <output line>
emit_row() {
    local csv=$1 platform=$2 version=$3 dataset=$4 workers=$5 nodes=$6 rep=$7 line=$8
    local ms acc
    ms=$(printf '%s' "$line"  | parse_ms)
    acc=$(printf '%s' "$line" | parse_acc)
    if [ -z "$ms" ]; then
        echo "  WARN: no runtime parsed for $version/$dataset/w=$workers rep=$rep" >&2
        return
    fi
    echo "$platform,$version,$dataset,$workers,$nodes,$rep,$ms,$acc" >> "$csv"
    printf "  %-8s %-7s w=%-4s n=%-4s rep=%s  %6s ms  acc=%s%%\n" \
           "$version" "$dataset" "$workers" "$nodes" "$rep" "$ms" "$acc"
}
