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
| `slurm/` | Slurm scripts and submission driver for Athena, plus hickory variants and `collect_hickory_results.py` |
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

## 5. Run on hickory

hickory.cs.vcu.edu is a different cluster from Athena, with real constraints
that limit how much of the sweep is possible there. `slurm/knn_job_hickory.sh`
and `slurm/submit_all_hickory.sh` are hickory-specific copies of the Athena
scripts (`slurm/knn_job.sh` and `slurm/submit_all.sh` are untouched and stay
Athena-only); use whichever pair matches the cluster you're on. What follows
was measured directly on hickory, not assumed from Athena:

- **One node only.** `sinfo -N` / `sinfo -s` show hickory's only partition
  (`gpu`, not `cpu-large`) contains exactly one node, `hickory` itself. There
  is no second node to place a multi-node MPI run on, so the `--nodes=2`
  comparison from the Athena sweep does not exist on hickory;
  `submit_all_hickory.sh` drops it entirely rather than skipping it at
  worker count 1.
- **Fewer usable CPUs than they look.** The node reports 24 CPUs, but
  `scontrol show node hickory` shows `CoreSpecCount=6` reserved outside
  Slurm's scheduler and `CfgTRES=cpu=18`, so only 18 cores are actually
  schedulable, shared with whatever else is running on the box at the time.
- **A low per-job CPU ceiling.** `sacctmgr show qos` shows QOS `short` caps a
  single job at `MaxCPUs=6` and QOS `long` (this account's default QOS) caps
  it at 3. `knn_job_hickory.sh` requests `--qos=short` explicitly so worker
  counts up to 6 are usable, and `submit_all_hickory.sh` defaults
  `WORKER_COUNTS` to `1 2 4 6` instead of Athena's `1 2 4 8 16 32 64 128`.
- **A tight submission cap.** `sacctmgr show assoc user=$USER` shows this
  account limited to `MaxSubmitJobs=2` / `MaxJobs=2`, i.e. at most 2 jobs may
  sit in the queue (pending or running) at once. Athena's `submit_all.sh`
  submits the whole sweep up front and lets Slurm queue the rest;
  `submit_all_hickory.sh` cannot do that (extra `sbatch` calls are rejected
  outright), so it submits one job at a time and polls `squeue -u $USER`,
  submitting the next job only once a slot frees up. This makes the hickory
  sweep much slower wall-clock-wise than the Athena one, even though each
  individual job is short.
- **GPU GRES is mandatory even for CPU-only jobs.** hickory's partition is a
  GPU partition with a Slurm job-submit policy that rejects any job with no
  GRES request. `knn_job_hickory.sh` requests the smallest GPU type at count
  1 (`--gres=gpu:40g:1`) purely to satisfy that policy; none of these four
  programs touch the GPU.
- **Different MPI module name.** `module avail mpi` lists `openmpi/4.1` on
  hickory, not Athena's `mpi/openmpi-4.1.6`.

```bash
./slurm/submit_all_hickory.sh
REPS=1 DATASETS="small" WORKER_COUNTS="1" ./slurm/submit_all_hickory.sh   # quick smoke test
```

Logs land in `slurm/logs/` with a `_hickory` suffix (e.g.
`threaded_small_4_rep1_hickory.log`) so they can sit alongside any Athena
logs without colliding. Once the sweep finishes, build `results/hickory.csv`
from those logs with:

```bash
python3 slurm/collect_hickory_results.py
```

which writes the same 8-column format `bench_common.sh` writes for maple, so
it drops straight into `analyze_results.py`:

```bash
python3 analyze_results.py --inputs results/maple.csv results/hickory.csv
```

## 6. Compute speedup and averages

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
directly for the report tables. hickory does have such a collector,
`slurm/collect_hickory_results.py` (see section 5), since its `.log` files
are named consistently enough (`_hickory` suffix) to parse automatically.

## Notes

- k defaults to 3 everywhere, matching the assignment. Pass a different `k`
  as the third command-line argument to any binary, or via the `K`
  environment variable to the scripts above.
- `serial.cpp` and the parser in `libarff/` are unmodified from the provided
  template.
- Develop and debug on maple; only run the full Athena sweep once
  `verify_accuracy.sh` passes cleanly on maple.
