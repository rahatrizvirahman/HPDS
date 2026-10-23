#!/usr/bin/env python3
"""Independently audit an MPI+CUDA scaling directory (from benchmark_mpi_cuda.py) against its logs and predictions.

Usage: audit_mpi_cuda_results.py DIRECTORY [JOBID ...]. LAYOUTS (default 1x1 1x2 1x4) and GPU_NAME (default H100) describe the
expected sweep. Slurm accounting is checked only when job IDs are given.
"""
import csv
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

DATASETS = ("small", "medium", "large")
LAYOUTS = os.environ.get("LAYOUTS", "1x1 1x2 1x4").split()
GPU_NAME = os.environ.get("GPU_NAME", "H100")
QUERIES = {"small": 160, "medium": 740, "large": 3436}


def main(directory, jobs):
    directory = Path(directory)
    out = []
    rows = list(csv.DictReader((directory / "raw.csv").open()))
    configs = [("serial", "-"), ("cuda", "single")] + [("mpi_cuda", layout) for layout in LAYOUTS]
    expected = 3 * len(DATASETS) * len(configs)
    assert len(rows) == expected, f"expected {expected} rows, found {len(rows)}"
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["program"], row["layout"]), []).append(row)
    assert len(groups) == len(DATASETS) * len(configs) and all(len(v) == 3 for v in groups.values())
    out.append(f"raw.csv rows {len(rows)} in {len(groups)} groups with 3 repetitions each")
    for row in rows:
        stem = f"{row['dataset']}-{row['program']}-{row['layout']}-r{row['rep']}"
        reference = (directory / f"{row['dataset']}-serial---r1.predictions.txt").read_bytes()
        assert (directory / f"{stem}.predictions.txt").read_bytes() == reference, f"prediction mismatch {stem}"
        text = (directory / f"{stem}.log").read_text()
        records = [l for l in text.splitlines() if l.startswith("RESULT ")]
        assert len(records) == 1, stem
        record = dict(f.split("=", 1) for f in records[0].split()[1:])
        assert record["end_to_end_ms"] == row["end_to_end_ms"] and record["accuracy"] == row["accuracy"], stem
        assert record.get("kernel_ms", "") == row["kernel_ms"] and row["predictions_match"] == "yes", stem
        if row["program"] != "serial":
            assert GPU_NAME in text, stem
        if row["program"] == "mpi_cuda":
            ranks = int(row["gpus"])
            assert int(record["mpi_ranks"]) == ranks and int(record["nodes"]) == int(row["nodes"]), stem
            rank_lines = [l for l in text.splitlines() if l.startswith("RANK ")]
            assert len(rank_lines) == ranks, stem
            devices = {(re.search(r"host=(\S+)", l).group(1), re.search(r"pci=(\S+)", l).group(1)) for l in rank_lines}
            assert len(devices) == ranks, f"ranks share a GPU in {stem}"
            assert sum(int(re.search(r"queries=(\d+)", l).group(1)) for l in rank_lines) == QUERIES[row["dataset"]], stem
    out.append(f"All {len(rows)} prediction files are byte-identical to serial and CSV rows equal their logs")
    out.append("Every MPI run used distinct GPUs per rank and its ranks covered every test query exactly once")
    summary = {(r["dataset"], r["program"], r["layout"]): r for r in csv.DictReader((directory / "summary.csv").open())}
    out.append("dataset program layout gpus mean_ms min max speedup_vs_1_mpi_gpu efficiency")
    for dataset in DATASETS:
        def mean(program, layout):
            return statistics.mean(float(r["end_to_end_ms"]) for r in groups[(dataset, program, layout)])
        serial = mean("serial", "-")
        for program, layout in configs:
            t = [float(r["end_to_end_ms"]) for r in groups[(dataset, program, layout)]]
            m = statistics.mean(t)
            claimed = summary[(dataset, program, layout)]
            for key, value in (("mean_ms", m), ("min_ms", min(t)), ("max_ms", max(t)), ("serial_speedup", serial / m)):
                assert abs(float(claimed[key]) - value) <= 1e-9 * max(1, abs(value)), (dataset, program, layout, key)
            scaling = ""
            if program == "mpi_cuda" and "1x1" in LAYOUTS:
                gpus = int(claimed["gpus"])
                scaling = mean("mpi_cuda", "1x1") / m
                assert abs(float(claimed["speedup_vs_1_mpi_gpu"]) - scaling) <= 1e-9 * max(1, scaling)
                assert abs(float(claimed["efficiency_vs_1_mpi_gpu"]) - scaling / gpus) <= 1e-9
            out.append(f"{dataset} {program} {layout} {claimed['gpus']} {m:.6f} {min(t):.6f} {max(t):.6f} "
                       f"{scaling if scaling == '' else format(scaling, '.4f')} "
                       f"{scaling if scaling == '' else format(scaling / int(claimed['gpus']), '.4f')}")
    out.append("summary.csv means, minima, maxima, speedups and efficiencies match independent recomputation")
    if jobs:
        accounting = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs), "--format=JobID,State,ExitCode"],
                                    text=True, stdout=subprocess.PIPE, check=True).stdout
        for line in accounting.splitlines():
            job, state, code = line.split("|")
            assert state == "COMPLETED" and code == "0:0", line
            out.append(f"job {job} {state} {code}")
    out.append("AUDIT PASSED")
    text = "\n".join(out) + "\n"
    (directory.parent / f"audit-{directory.name}.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
