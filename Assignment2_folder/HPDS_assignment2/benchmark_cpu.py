#!/usr/bin/env python3
"""Measure the CPU programs (serial, pthreads, OpenMP, MPI) and compare every prediction file with serial.

Environment (defaults in brackets). K [3], REPS [3], DATASETS [small medium large], WORKERS [1 2 4 8 16],
MPI_LAUNCH [mpirun -n {ranks}], RESULTS_ROOT [results], RUN_ID. Each dataset gets three serial runs and three runs
for every program and worker count. The timing boundary is the same as the GPU programs (classification after parsing).
"""
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import socket
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import benchmark_baseline as base  # noqa: E402


def main():
    k = base.positive(os.environ.get("K", "3"))
    repetitions = base.positive(os.environ.get("REPS", "3"))
    datasets = os.environ.get("DATASETS", "small medium large").split()
    workers = [base.positive(v) for v in os.environ.get("WORKERS", "1 2 4 8 16").split()]
    template = os.environ.get("MPI_LAUNCH", "mpirun -n {ranks}")
    if not datasets or not workers or any(d not in ("small", "medium", "large") for d in datasets):
        raise ValueError("Select known datasets and worker counts")
    output = ROOT / os.environ.get("RESULTS_ROOT", "results")
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = os.environ.get("RUN_ID", "")
    destination = Path(tempfile.mkdtemp(prefix=f"cpu-benchmark-{stamp}-{run_id + '-' if run_id else ''}", dir=output))
    print(f"Results directory {destination}", flush=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    (destination / "metadata.txt").write_text(
        f"host={socket.gethostname()}\ncommit={revision.stdout.strip()}\nslurm_job={os.environ.get('SLURM_JOB_ID', '')}\n"
        f"run_id={run_id}\nk={k}\nrepetitions={repetitions}\nworkers={workers}\nmpi_launch={template}\nwarmup=none\n"
        "Each measurement launches a fresh process. MPI time is the maximum over ranks. Preserve the Slurm log for the CPU model.\n")
    fields = ["host", "dataset", "k", "program", "workers", "rep", "end_to_end_ms", "accuracy", "predictions_match"]
    configurations = [("serial", 1)] + [(p, w) for p in ("pthreads", "openmp", "mpi") for w in workers]
    rows = []
    with (destination / "raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset in datasets:
            inputs = [f"datasets/{dataset}-train.arff", f"datasets/{dataset}-test.arff", str(k)]
            reference = None
            for program, w in configurations:
                for rep in range(1, repetitions + 1):
                    stem = f"{dataset}-{program}-w{w}-r{rep}"
                    prediction_file = destination / f"{stem}.predictions.txt"
                    if program == "serial":
                        command = ["./serial"] + inputs
                    elif program == "pthreads":
                        command = ["./threaded"] + inputs + [str(w)]
                    elif program == "openmp":
                        command = ["./openmp"] + inputs + [str(w)]
                    else:
                        command = shlex.split(template.format(ranks=w)) + ["./mpi"] + inputs
                    record = base.run(command + [str(prediction_file)], destination / f"{stem}.log")
                    if record["strategy"] != program or (program != "serial" and int(record["workers"]) != w):
                        raise ValueError(f"Unexpected program or worker count in {stem}")
                    predictions = prediction_file.read_text()
                    if not predictions.strip():
                        raise ValueError("Empty predictions")
                    if reference is None:
                        reference = predictions
                    if predictions != reference:
                        raise ValueError(f"Prediction mismatch in {stem}, raw files retained")
                    row = dict(host=socket.gethostname(), dataset=dataset, k=k, program=program, workers=w, rep=rep,
                               end_to_end_ms=record["end_to_end_ms"], accuracy=record["accuracy"], predictions_match="yes")
                    writer.writerow(row)
                    stream.flush()
                    rows.append(row)
    summary_fields = ["dataset", "program", "workers", "runs", "mean_ms", "min_ms", "max_ms", "speedup_vs_serial",
                      "speedup_vs_1_worker", "efficiency_vs_1_worker"]
    with (destination / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for dataset in datasets:
            def times(program, w):
                return [float(r["end_to_end_ms"]) for r in rows if (r["dataset"], r["program"], int(r["workers"])) == (dataset, program, w)]
            serial_mean = statistics.mean(times("serial", 1))
            for program, w in configurations:
                t = times(program, w)
                mean = statistics.mean(t)
                own_one = statistics.mean(times(program, 1)) if program != "serial" and 1 in workers else None
                scaling = own_one / mean if own_one else ""
                writer.writerow(dict(dataset=dataset, program=program, workers=w, runs=len(t), mean_ms=mean, min_ms=min(t),
                                     max_ms=max(t), speedup_vs_serial=serial_mean / mean, speedup_vs_1_worker=scaling,
                                     efficiency_vs_1_worker=scaling / w if scaling != "" else ""))
    print(f"All CPU predictions matched. Results saved in {destination}")


if __name__ == "__main__":
    main()
