#!/usr/bin/env python3
"""Builds results/hickory.csv from the .log files slurm/submit_all_hickory.sh
produces in slurm/logs/, in the same 8-column format bench_common.sh writes
for maple (see its csv_header line), so it can be passed straight to
analyze_results.py alongside results/maple.csv.

Run this yourself after slurm/submit_all_hickory.sh's jobs have finished. It
does not submit or wait for anything, it only reads whatever logs already
exist.

Usage:
    python3 slurm/collect_hickory_results.py
    python3 slurm/collect_hickory_results.py --logs-dir slurm/logs --out results/hickory.csv
    python3 analyze_results.py --inputs results/maple.csv results/hickory.csv
"""

import argparse
import csv
import re
from pathlib import Path

FILENAME_RE = re.compile(
    r"^(?P<version>serial|threaded|openmp|mpi_1node)_"
    r"(?P<dataset>small|medium|large)"
    r"(?:_(?P<workers>\d+))?_rep(?P<rep>\d+)_hickory\.log$"
)
NODES_RE = re.compile(r"nodes used:\s*(\d+)")
TIME_RE = re.compile(r"required (\d+) ms")
ACC_RE = re.compile(r"Accuracy was ([\d.]+)%")


def parse_log(path):
    text = path.read_text()
    nodes_m = NODES_RE.search(text)
    time_m = TIME_RE.search(text)
    acc_m = ACC_RE.search(text)
    if not time_m:
        return None
    return {
        "nodes": nodes_m.group(1) if nodes_m else "1",
        "time_ms": time_m.group(1),
        "accuracy": acc_m.group(1) if acc_m else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="slurm/logs")
    ap.add_argument("--out", default="results/hickory.csv")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    rows = []
    skipped = []
    for path in sorted(logs_dir.glob("*_hickory.log")):
        m = FILENAME_RE.match(path.name)
        if not m:
            skipped.append((path.name, "filename didn't match the expected pattern"))
            continue
        parsed = parse_log(path)
        if parsed is None:
            skipped.append((path.name, "no runtime line found in log"))
            continue

        version = "mpi" if m.group("version") == "mpi_1node" else m.group("version")
        workers = m.group("workers") or "1"

        rows.append({
            "platform": "hickory",
            "version": version,
            "dataset": m.group("dataset"),
            "workers": workers,
            "nodes": parsed["nodes"],
            "rep": m.group("rep"),
            "time_ms": parsed["time_ms"],
            "accuracy": parsed["accuracy"],
        })

    for name, reason in skipped:
        print(f"  WARN: skipping {name} ({reason})")

    if not rows:
        print(f"No hickory logs found in {logs_dir}/. Run slurm/submit_all_hickory.sh first.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["platform", "version", "dataset", "workers", "nodes", "rep", "time_ms", "accuracy"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows, {len(skipped)} skipped)")


if __name__ == "__main__":
    main()
