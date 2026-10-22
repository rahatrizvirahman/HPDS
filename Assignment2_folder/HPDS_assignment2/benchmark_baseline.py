#!/usr/bin/env python3
"""Record baseline measurements only after exact prediction comparisons pass."""
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent


def results_root():
    """Output root from RESULTS_ROOT, relative to this directory unless absolute. Default is results."""
    return (ROOT / os.environ.get("RESULTS_ROOT", "results")).resolve()


def results_root():
    """Output root from RESULTS_ROOT, relative to this directory unless absolute. Defaults to results."""
    return ROOT / os.environ.get("RESULTS_ROOT", "results")


def positive(value):
    result = int(value)
    if result < 1:
        raise ValueError("Expected a positive integer")
    return result


def run(command, log):
    process = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    log.write_text(process.stdout)
    print(process.stdout, end="", flush=True)
    process.check_returncode()
    records = [line for line in process.stdout.splitlines() if line.startswith("RESULT ")]
    if len(records) != 1:
        raise ValueError("Expected exactly one RESULT record")
    record = dict(field.split("=", 1) for field in records[0].split()[1:])
    for key in ("end_to_end_ms", "accuracy"):
        if not math.isfinite(float(record[key])):
            raise ValueError("Non-finite measurement")
    if float(record["end_to_end_ms"]) <= 0 or not 0 <= float(record["accuracy"]) <= 100:
        raise ValueError("Invalid measurement")
    if "kernel_ms" in record and (not math.isfinite(float(record["kernel_ms"])) or float(record["kernel_ms"]) < 0):
        raise ValueError("Invalid kernel measurement")
    return record


def main():
    k = positive(os.environ.get("K", "3"))
    repetitions = positive(os.environ.get("REPS", "3"))
    datasets = os.environ.get("DATASETS", "small medium large").split()
    blocks = [positive(v) for v in os.environ.get("BLOCK_SIZES", "64 128 256").split()]
    if not datasets or not blocks or any(d not in ("small", "medium", "large") for d in datasets):
        raise ValueError("Select known datasets and at least one block size")
    output = results_root()
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = os.environ.get("RUN_ID", "")
    destination = Path(tempfile.mkdtemp(prefix=f"baseline-{stamp}-{run_id + '-' if run_id else ''}", dir=output))
    print(f"Results directory {destination}", flush=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    (destination / "metadata.txt").write_text(
        f"host={socket.gethostname()}\ncommit={revision.stdout.strip()}\n"
        f"slurm_job={os.environ.get('SLURM_JOB_ID', '')}\nrun_id={run_id}\n"
        f"k={k}\nrepetitions={repetitions}\nblock_sizes={blocks}\n"
        "warmup=none\nEach measurement launches a fresh process.\n"
        "Preserve the Slurm log for GPU model, compiler versions, and build flags.\n")
    fields = ["host", "dataset", "k", "strategy", "block_size", "rep",
              "end_to_end_ms", "kernel_ms", "accuracy", "predictions_match"]
    rows = []
    with (destination / "raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset in datasets:
            inputs = [f"datasets/{dataset}-train.arff", f"datasets/{dataset}-test.arff", str(k)]
            reference = None
            for strategy, block in [("serial", 0)] + [("global", b) for b in blocks]:
                for rep in range(1, repetitions + 1):
                    stem = f"{dataset}-{strategy}-b{block}-r{rep}"
                    prediction_file = destination / f"{stem}.predictions.txt"
                    command = ["./serial"] if strategy == "serial" else ["./cuda"]
                    command += inputs + ([] if strategy == "serial" else [str(block)])
                    record = run(command + [str(prediction_file)], destination / f"{stem}.log")
                    if record["strategy"] != strategy:
                        raise ValueError("Unexpected program strategy")
                    predictions = prediction_file.read_text()
                    if not predictions.strip():
                        raise ValueError("Empty predictions")
                    if reference is None:
                        reference = predictions
                    if predictions != reference:
                        raise ValueError(f"Prediction mismatch in {stem}, raw files retained")
                    row = dict(host=socket.gethostname(), dataset=dataset, k=k, strategy=strategy,
                               block_size=block, rep=rep, end_to_end_ms=record["end_to_end_ms"],
                               kernel_ms=record.get("kernel_ms", ""), accuracy=record["accuracy"],
                               predictions_match="yes")
                    writer.writerow(row)
                    stream.flush()
                    rows.append(row)
    summary_fields = ["dataset", "strategy", "block_size", "runs", "mean_ms", "min_ms", "max_ms", "serial_speedup"]
    with (destination / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for dataset in datasets:
            serial_mean = statistics.mean(float(r["end_to_end_ms"]) for r in rows
                                          if r["dataset"] == dataset and r["strategy"] == "serial")
            for strategy, block in [("serial", 0)] + [("global", b) for b in blocks]:
                times = [float(r["end_to_end_ms"]) for r in rows if r["dataset"] == dataset
                         and r["strategy"] == strategy and r["block_size"] == block]
                mean = statistics.mean(times)
                writer.writerow(dict(dataset=dataset, strategy=strategy, block_size=block,
                                     runs=len(times), mean_ms=mean, min_ms=min(times), max_ms=max(times),
                                     serial_speedup=serial_mean / mean))
    print(f"All baseline predictions matched. Results saved in {destination}")


if __name__ == "__main__":
    main()
