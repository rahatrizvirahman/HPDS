#!/usr/bin/env python3
"""Independently audit a CPU benchmark directory against its logs and predictions.

Usage: audit_cpu_results.py DIRECTORY JOBID [JOBID ...]. WORKERS (default 1 2 4 8 16) names the expected worker counts.
"""
import csv
import os
from pathlib import Path
import statistics
import subprocess
import sys

DATASETS = ("small", "medium", "large")
WORKERS = [int(v) for v in os.environ.get("WORKERS", "1 2 4 8 16").split()]
CONFIGS = [("serial", 1)] + [(p, w) for p in ("pthreads", "openmp", "mpi") for w in WORKERS]


def main(directory, jobs):
    directory = Path(directory)
    out = []
    rows = list(csv.DictReader((directory / "raw.csv").open()))
    expected = 3 * len(DATASETS) * len(CONFIGS)
    assert len(rows) == expected, f"expected {expected} rows, found {len(rows)}"
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["program"], int(row["workers"])), []).append(row)
    assert len(groups) == len(DATASETS) * len(CONFIGS) and all(len(v) == 3 for v in groups.values())
    assert all(sorted(int(r["rep"]) for r in v) == [1, 2, 3] for v in groups.values())
    out.append(f"raw.csv rows {len(rows)} in {len(groups)} dataset/program/worker groups with 3 repetitions each")
    for row in rows:
        stem = f"{row['dataset']}-{row['program']}-w{row['workers']}-r{row['rep']}"
        reference = (directory / f"{row['dataset']}-serial-w1-r1.predictions.txt").read_bytes()
        assert (directory / f"{stem}.predictions.txt").read_bytes() == reference, f"prediction mismatch {stem}"
        records = [l for l in (directory / f"{stem}.log").read_text().splitlines() if l.startswith("RESULT ")]
        assert len(records) == 1, stem
        record = dict(f.split("=", 1) for f in records[0].split()[1:])
        assert record["strategy"] == row["program"] and record["end_to_end_ms"] == row["end_to_end_ms"], stem
        assert record["accuracy"] == row["accuracy"] and row["predictions_match"] == "yes" and row["k"] == "3", stem
        if row["program"] != "serial":
            assert int(record["workers"]) == int(row["workers"]), stem
    out.append(f"All {len(rows)} prediction files are byte-identical to the dataset's serial reference")
    out.append(f"All {len(rows)} CSV rows equal the RESULT records in their individual logs")
    summary = {(r["dataset"], r["program"], int(r["workers"])): r for r in csv.DictReader((directory / "summary.csv").open())}
    out.append("dataset program workers runs mean_ms min max speedup_vs_serial")
    for dataset in DATASETS:
        serial = statistics.mean(float(r["end_to_end_ms"]) for r in groups[(dataset, "serial", 1)])
        for program, w in CONFIGS:
            t = [float(r["end_to_end_ms"]) for r in groups[(dataset, program, w)]]
            mean = statistics.mean(t)
            claimed = summary[(dataset, program, w)]
            for key, value in (("mean_ms", mean), ("min_ms", min(t)), ("max_ms", max(t)), ("speedup_vs_serial", serial / mean)):
                assert abs(float(claimed[key]) - value) <= 1e-9 * max(1, abs(value)), (dataset, program, w, key)
            if program != "serial" and 1 in WORKERS:
                one = statistics.mean(float(r["end_to_end_ms"]) for r in groups[(dataset, program, 1)])
                assert abs(float(claimed["speedup_vs_1_worker"]) - one / mean) <= 1e-9 * max(1, one / mean)
                assert abs(float(claimed["efficiency_vs_1_worker"]) - one / mean / w) <= 1e-9
            out.append(f"{dataset} {program} {w} 3 {mean:.6f} {min(t):.6f} {max(t):.6f} {serial / mean:.4f}")
    out.append("summary.csv mean, minimum, maximum, speedup and efficiency values match independent recomputation")
    accounting = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs), "--format=JobID,State,ExitCode"],
                                text=True, stdout=subprocess.PIPE, check=True).stdout
    for line in accounting.splitlines():
        job, state, code = line.split("|")
        assert state == "COMPLETED" and code == "0:0", line
        out.append(f"job {job} {state} {code}")
    out.append("AUDIT PASSED")
    text = "\n".join(out) + "\n"
    (directory.parent / "audit.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
