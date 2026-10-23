#!/usr/bin/env python3
"""Independently audit an Athena benchmark directory against its logs and predictions."""
import csv
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

DATASETS = ("small", "medium", "large")
# STRATEGIES (default global) names the CUDA strategies the sweep was expected to contain.
STRATEGIES = os.environ.get("STRATEGIES", "global").split()
# GPU_NAME (default A100) is the text every CUDA log must contain, so mixed hardware is rejected.
GPU_NAME = os.environ.get("GPU_NAME", "A100")
CONFIGS = [("serial", 0)] + [(s, b) for s in STRATEGIES for b in (64, 128, 256)]


def main(directory, jobs):
    directory = Path(directory)
    out = []
    rows = list(csv.DictReader((directory / "raw.csv").open()))
    expected_groups = len(DATASETS) * len(CONFIGS)
    expected_rows = 3 * expected_groups
    assert len(rows) == expected_rows, f"expected {expected_rows} rows, found {len(rows)}"
    out.append(f"raw.csv rows {len(rows)}")
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["strategy"], int(row["block_size"])), []).append(row)
    assert len(groups) == expected_groups and all(len(v) == 3 for v in groups.values())
    assert sum(g[1] == "serial" for g in groups) == 3
    assert all(sum(g[1] == s for g in groups) == 9 for s in STRATEGIES)
    out.append(f"{expected_groups} dataset/strategy/block groups with 3 repetitions each "
               f"(9 serial and {27 * len(STRATEGIES)} CUDA rows)")
    assert all(sorted(int(r["rep"]) for r in v) == [1, 2, 3] for v in groups.values())
    # Every prediction file against the first serial file, and every CSV value against its process log.
    for row in rows:
        stem = f"{row['dataset']}-{row['strategy']}-b{row['block_size']}-r{row['rep']}"
        reference = (directory / f"{row['dataset']}-serial-b0-r1.predictions.txt").read_bytes()
        assert (directory / f"{stem}.predictions.txt").read_bytes() == reference, f"prediction mismatch {stem}"
        text = (directory / f"{stem}.log").read_text()
        records = [l for l in text.splitlines() if l.startswith("RESULT ")]
        assert len(records) == 1, stem
        record = dict(f.split("=", 1) for f in records[0].split()[1:])
        assert record["strategy"] == row["strategy"], stem
        assert record["end_to_end_ms"] == row["end_to_end_ms"], stem
        assert record["accuracy"] == row["accuracy"], stem
        assert record.get("kernel_ms", "") == row["kernel_ms"], stem
        assert row["predictions_match"] == "yes" and row["k"] == "3", stem
        if row["strategy"] != "serial":
            assert GPU_NAME in text, stem
    out.append(f"All {len(rows)} prediction files are byte-identical to the dataset's serial reference")
    out.append(f"All {len(rows)} CSV rows equal the RESULT records in their individual logs")
    # Recompute summaries.
    summary = {(r["dataset"], r["strategy"], int(r["block_size"])): r
               for r in csv.DictReader((directory / "summary.csv").open())}
    kernel_rows = [["dataset", "strategy", "block_size", "runs", "mean_kernel_ms", "min_kernel_ms", "max_kernel_ms",
                    "mean_end_to_end_ms", "kernel_fraction_of_end_to_end"]]
    out.append("dataset strategy block runs mean_e2e min max mean_kernel speedup")
    for dataset in DATASETS:
        serial_mean = statistics.mean(float(r["end_to_end_ms"]) for r in groups[(dataset, "serial", 0)])
        for strategy, block in CONFIGS:
            e2e = [float(r["end_to_end_ms"]) for r in groups[(dataset, strategy, block)]]
            mean = statistics.mean(e2e)
            claimed = summary[(dataset, strategy, block)]
            for key, value in (("mean_ms", mean), ("min_ms", min(e2e)), ("max_ms", max(e2e)),
                               ("serial_speedup", serial_mean / mean)):
                assert abs(float(claimed[key]) - value) <= 1e-9 * max(1, abs(value)), (dataset, strategy, block, key)
            assert int(claimed["runs"]) == 3
            kmean = ""
            if strategy != "serial":
                kernel = [float(r["kernel_ms"]) for r in groups[(dataset, strategy, block)]]
                assert all(k < e for k, e in zip(kernel, e2e))
                kmean = statistics.mean(kernel)
                kernel_rows.append([dataset, strategy, block, 3, f"{kmean:.6f}", f"{min(kernel):.6f}", f"{max(kernel):.6f}",
                                    f"{mean:.6f}", f"{kmean / mean:.6f}"])
            out.append(f"{dataset} {strategy} {block} 3 {mean:.6f} {min(e2e):.6f} {max(e2e):.6f} "
                       f"{kmean if kmean == '' else format(kmean, '.6f')} {serial_mean / mean:.4f}")
    with (directory / "kernel-summary.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows(kernel_rows)
    out.append("summary.csv mean, minimum, maximum, run counts and speedups match independent recomputation")
    # Slurm states and exit codes.
    accounting = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs),
                                 "--format=JobID,State,ExitCode"], text=True, stdout=subprocess.PIPE, check=True).stdout
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
