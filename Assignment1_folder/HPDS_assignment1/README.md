# HPDS Assignment 1: KNN with threads, OpenMP, and MPI

Serial, pthreads, OpenMP, and MPI implementations of k-nearest-neighbor
classification, plus the scripts used to benchmark them on `maple.cs.vcu.edu`
and `athena.hprc.vcu.edu`.

All four programs partition work the same way: each worker (thread, or MPI
rank) takes a contiguous block of **test** instances and scans the full
training set for each one, exactly as `serial.cpp` does. This keeps every
worker's arithmetic identical to the serial version in the same order, so
predictions match the serial output exactly at every worker count. See
`Assignment1_folder/Explanations/` for the full reasoning behind this and
every other design decision.

## Files

| Path | Purpose |
|---|---|
| `serial.cpp`, `threaded.cpp`, `openmp.cpp`, `mpi.cpp` | The four implementations |
| `libarff/` | Provided ARFF file parser, unmodified |
| `datasets/` | small / medium / large train and test ARFF files |
| `Makefile` | Builds all four binaries |
| `verify_accuracy.sh` | Confirms every implementation matches the serial accuracy |
| `run_experiments.sh`, `bench_common.sh` | Benchmark sweep for maple |
| `slurm/` | Slurm scripts and submission driver for Athena |
| `analyze_results.py` | Averages repetitions, computes speedup and parallel-fraction tables/figures |
| `results/` | CSVs and generated figure data (created by the scripts above) |

## 1. Build

```bash
make            # builds serial, threaded, openmp, mpi
make clean      # removes the four binaries
```

## 2. Verify correctness (do this before any timed run)

```bash
./verify_accuracy.sh                    # k=3, worker counts 1 2 4 8 16
./verify_accuracy.sh 3 1 2 4 8 16 32 64 128   # full worker-count sweep
```

This runs every implementation against every dataset and checks the reported
accuracy against `./serial`'s accuracy. It must print `ALL ACCURACIES MATCH
SERIAL` before you trust any timing result. If it does not, the code has a
race or an indexing bug: fix that first, since the assignment is graded in
part on exact accuracy match.

## 3. Run the benchmark sweep on maple

```bash
./run_experiments.sh
```

Defaults to k=3, 3 repetitions, all three datasets, worker counts
1 2 4 8 16 32 64 128. Override any of these with environment variables:

```bash
REPS=1 DATASETS="small" WORKER_COUNTS="1 2 4" ./run_experiments.sh
```

Writes `results/maple.csv` with one row per (version, dataset, workers, rep).

## 4. Run on Athena

Copy this whole folder (code, `datasets/`, `Makefile`, `slurm/`) to Athena and
`make` there. Every run goes through the one script `slurm/knn_job.sh`; which
program runs and how many CPUs/nodes it gets are chosen by the `sbatch` flags,
not by separate scripts, since a single Slurm job's resource reservation is
fixed for its whole run and can't grow from 1 CPU to 128 CPUs partway through.
That's also why "run the whole sweep" here means submitting many small jobs,
one per configuration, rather than one big job that loops internally.

```bash
./slurm/submit_all.sh
```

submits the entire sweep for you: every dataset, every worker count 1 through
128, 3 repetitions each, and both the single-node and multi-node MPI variants
(~288 jobs with the defaults). It returns immediately after submitting; it
does not wait for the jobs to finish. Track progress with `squeue -u $USER`
and let the queue drain before moving on to analysis. Shrink it for a quick
check with the same override variables as `run_experiments.sh`:

```bash
REPS=1 DATASETS="small" WORKER_COUNTS="1 2 4" ./slurm/submit_all.sh
```

Each `.log` file lands in `slurm/logs/` and contains the same "required NNN
ms ... Accuracy was NN.NN%" line the program prints on maple, plus a "nodes
used: N" line so you can confirm the 1-node vs multi-node MPI runs actually
landed on different node counts.

`knn_job.sh` targets the `cpu-large` partition, which only contains AMD nodes
of similar age, so runtimes across reps stay comparable instead of drifting
because two reps happened to land on nodes of very different ages. If you
ever call `sbatch` on it directly instead of through `submit_all.sh`, don't
pass `--ntasks` and `--ntasks-per-node` in the same command; use one or the
other. `--nodes` with `--ntasks` is fine, and is what `submit_all.sh` uses.

## 5. Compute speedup and averages

Pull the runtime out of each `.log` file and average the 3 repetitions.
This can be done by hand from the printed lines, or with a short `grep`:

```bash
grep -h "required" slurm/logs/threaded_small_8_rep*.log
```

`analyze_results.py` (optional) automates the averaging plus the absolute
and relative speedup and Karp-Flatt parallel-fraction numbers described in
`Assignment1_folder/Explanations/00-overview-and-decomposition.md`, if you
would rather not compute those by hand for every row of the report tables.
It reads `results/maple.csv` directly, since `run_experiments.sh` writes it.
There is no script that builds a matching `results/athena.csv` from the
Slurm `.log` files (`submit_all.sh` only submits jobs, it does not collect
their results into a CSV), so for Athena either add
rows to a CSV by hand in the same 8-column format `bench_common.sh` writes
for maple (see its `csv_header` line), or just average the `.log` numbers
directly for the report tables.

## Notes

- k defaults to 3 everywhere, matching the assignment. Pass a different `k`
  as the third command-line argument to any binary, or via the `K`
  environment variable to the scripts above.
- `serial.cpp` and the parser in `libarff/` are unmodified from the provided
  template.
- Develop and debug on maple; only run the full Athena sweep once
  `verify_accuracy.sh` passes cleanly on maple.
