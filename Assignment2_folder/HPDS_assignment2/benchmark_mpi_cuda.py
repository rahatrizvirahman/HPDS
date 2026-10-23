#!/usr/bin/env python3
"""Measure MPI+CUDA scaling. Every prediction file is compared with serial before a summary is written.

Environment (defaults in brackets). K [3], REPS [3], DATASETS [small medium large], BLOCK_SIZE [128],
STRATEGY [shared], TILE_ROWS [256], LAYOUTS [1x1 1x2 1x4] as NODESxRANKS_PER_NODE, MPI_LAUNCH, RESULTS_ROOT [results], RUN_ID.
Each dataset gets three serial runs, three single-process ./cuda runs and three MPI+CUDA runs per layout.
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
import benchmark_baseline as base  # noqa: E402  (reuses the validated RESULT parser)


def results_root():
    return ROOT / os.environ.get("RESULTS_ROOT", "results")


def launcher(layout, template):
    nodes, per_node = (int(v) for v in layout.split("x"))
    return shlex.split(template.format(nodes=nodes, per_node=per_node, ranks=nodes * per_node)), nodes, nodes * per_node


def main():
    k = base.positive(os.environ.get("K", "3"))
    repetitions = base.positive(os.environ.get("REPS", "3"))
    block = base.positive(os.environ.get("BLOCK_SIZE", "128"))
    tile = base.positive(os.environ.get("TILE_ROWS", "256"))
    strategy = os.environ.get("STRATEGY", "shared")
    datasets = os.environ.get("DATASETS", "small medium large").split()
    layouts = os.environ.get("LAYOUTS", "1x1 1x2 1x4").split()
    template = os.environ.get("MPI_LAUNCH", "mpirun -n {ranks}")
    if strategy not in ("global", "shared") or not datasets or not layouts or any(d not in ("small", "medium", "large") for d in datasets):
        raise ValueError("Select known datasets, layouts and a global or shared strategy")
    flags = ["--strategy=shared", f"--tile={tile}"] if strategy == "shared" else []
    output = results_root()
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = os.environ.get("RUN_ID", "")
    destination = Path(tempfile.mkdtemp(prefix=f"mpi-benchmark-{stamp}-{run_id + '-' if run_id else ''}", dir=output))
    print(f"Results directory {destination}", flush=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    (destination / "metadata.txt").write_text(
        f"host={socket.gethostname()}\ncommit={revision.stdout.strip()}\nslurm_job={os.environ.get('SLURM_JOB_ID', '')}\n"
        f"run_id={run_id}\nk={k}\nrepetitions={repetitions}\nstrategy={strategy}\nblock_size={block}\ntile_rows={tile}\n"
        f"layouts={layouts}\nmpi_launch={template}\nwarmup=none\nEach measurement launches a fresh process or job step.\n"
        "MPI elapsed time is the maximum over ranks of the timed region. Preserve the Slurm log for hardware and build flags.\n")
    fields = ["host", "dataset", "k", "program", "layout", "gpus", "nodes", "strategy", "block_size", "rep",
              "end_to_end_ms", "kernel_ms", "accuracy", "predictions_match"]
    configurations = [("serial", "-", 1, 1), ("cuda", "single", 1, 1)] + [("mpi_cuda", layout, *launcher(layout, template)[1:]) for layout in layouts]
    rows = []
    with (destination / "raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset in datasets:
            inputs = [f"datasets/{dataset}-train.arff", f"datasets/{dataset}-test.arff", str(k)]
            reference = None
            for program, layout, nodes, gpus in configurations:
                for rep in range(1, repetitions + 1):
                    stem = f"{dataset}-{program}-{layout}-r{rep}"
                    prediction_file = destination / f"{stem}.predictions.txt"
                    if program == "serial":
                        command = ["./serial"] + inputs
                    elif program == "cuda":
                        command = ["./cuda"] + inputs + [str(block)] + flags
                    else:
                        command = launcher(layout, template)[0] + ["./mpi_cuda"] + inputs + [str(block)] + flags
                    record = base.run(command + [str(prediction_file)], destination / f"{stem}.log")
                    expected = "serial" if program == "serial" else strategy
                    if record["strategy"] != expected:
                        raise ValueError("Unexpected program strategy")
                    if program == "mpi_cuda" and (int(record["mpi_ranks"]) != gpus or int(record["nodes"]) != nodes):
                        raise ValueError(f"Layout {layout} ran with {record['mpi_ranks']} ranks on {record['nodes']} nodes")
                    predictions = prediction_file.read_text()
                    if not predictions.strip():
                        raise ValueError("Empty predictions")
                    if reference is None:
                        reference = predictions
                    if predictions != reference:
                        raise ValueError(f"Prediction mismatch in {stem}, raw files retained")
                    row = dict(host=socket.gethostname(), dataset=dataset, k=k, program=program, layout=layout, gpus=gpus,
                               nodes=nodes, strategy=expected, block_size=0 if program == "serial" else block, rep=rep,
                               end_to_end_ms=record["end_to_end_ms"], kernel_ms=record.get("kernel_ms", ""),
                               accuracy=record["accuracy"], predictions_match="yes")
                    writer.writerow(row)
                    stream.flush()
                    rows.append(row)
    summary_fields = ["dataset", "program", "layout", "gpus", "nodes", "runs", "mean_ms", "min_ms", "max_ms", "mean_kernel_ms",
                      "serial_speedup", "speedup_vs_1_mpi_gpu", "efficiency_vs_1_mpi_gpu"]
    with (destination / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for dataset in datasets:
            def mean_of(program, layout, key="end_to_end_ms"):
                return statistics.mean(float(r[key]) for r in rows if (r["dataset"], r["program"], r["layout"]) == (dataset, program, layout))
            serial_mean = mean_of("serial", "-")
            one_gpu = mean_of("mpi_cuda", "1x1") if "1x1" in layouts else None
            for program, layout, nodes, gpus in configurations:
                times = [float(r["end_to_end_ms"]) for r in rows if (r["dataset"], r["program"], r["layout"]) == (dataset, program, layout)]
                mean = statistics.mean(times)
                kernel = "" if program == "serial" else mean_of(program, layout, "kernel_ms")
                scaling = one_gpu / mean if program == "mpi_cuda" and one_gpu else ""
                writer.writerow(dict(dataset=dataset, program=program, layout=layout, gpus=gpus, nodes=nodes, runs=len(times),
                                     mean_ms=mean, min_ms=min(times), max_ms=max(times), mean_kernel_ms=kernel,
                                     serial_speedup=serial_mean / mean, speedup_vs_1_mpi_gpu=scaling,
                                     efficiency_vs_1_mpi_gpu=scaling / gpus if scaling != "" else ""))
    print(f"All MPI+CUDA predictions matched. Results saved in {destination}")


if __name__ == "__main__":
    main()
