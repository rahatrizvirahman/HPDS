# Assignment 2 CUDA baseline

This is the first implementation milestone. It contains a serial reference and a single-GPU CUDA baseline. The shared-memory strategy, MPI+CUDA implementation, full CPU comparison sweep, and Overleaf report remain planned work in [ASSIGNMENT2_PLAN.md](../ASSIGNMENT2_PLAN.md).

The supplied datasets and parser are unchanged. Historical Assignment 1 results, scripts, and report are preserved in [Reference_assignment1](../Reference_assignment1/). They are not Assignment 2 results.

## Clone and inspect each cluster

Run these commands on both Maple and Athena. If a checkout already exists, use `git pull --ff-only` from that checkout instead of cloning again.

```bash
git clone https://github.com/rahatrizvirahman/HPDS.git
cd HPDS/Assignment2_folder/HPDS_assignment2
mkdir -p results slurm/logs
bash slurm/inspect_cluster.sh 2>&1 | tee "results/cluster-$(hostname).txt"
```

Send the two cluster inspection files back before finalizing the GPU submission commands. The script queries Slurm and module availability without launching computation. A login node need not expose a GPU or have `nvcc` loaded. Missing GPU visibility on a login node is not a failed GPU test.

We need the GPU partition, allocation syntax, available CUDA/compiler modules, any account/QOS requirement, and the GPU counts allowed per job. The script prints readable account and QOS information where available. Administrator restrictions or site instructions may still need to supply missing limits.

## Build and first verification job

The Makefile builds `serial` and `cuda` by default. It requires a C++ compiler and NVIDIA's `nvcc`. It does not require MPI for this milestone. The Slurm job rebuilds both programs inside the allocation, records the compiler versions and GPU information, and runs correctness checks.

After confirming the cluster settings, load its actual compiler/CUDA modules and submit the job. The following is a template. Replace the partition and, if required by the site, the GPU request with the verified values. Add the actual account and QOS options if required.

```bash
# Run from HPDS/Assignment2_folder/HPDS_assignment2.
# Load the verified compiler and CUDA modules in this shell first.
mkdir -p slurm/logs
sbatch --partition=REPLACE_WITH_GPU_PARTITION --gres=gpu:1 \
  --output=slurm/logs/baseline-%j.log \
  slurm/baseline_job.sh verify
squeue -u "$USER"
```

`MODULES` can also specify a space-separated list of actual module names to load inside the job. No guessed cluster module names or GPU types are embedded in the script. The same job body supports either cluster through its submission options.

For an initial small-data check, export the selection before submitting.

```bash
export DATASETS=small BLOCK_SIZES='64 128 256'
# Submit the verified sbatch command above.
```

For the full verification run, unset those overrides or set all three datasets. GPU jobs must run through Slurm allocations, not directly on login nodes. A successful job prints `ALL PREDICTIONS MATCH SERIAL`. A build failure or mismatch produces a nonzero job exit and retains the Slurm log.

## Program interfaces

Within a GPU allocation, the individual interfaces are as follows.

```bash
./serial datasets/small-train.arff datasets/small-test.arff 3 serial-predictions.txt
./cuda datasets/small-train.arff datasets/small-test.arff 3 128 cuda-predictions.txt
```

The final prediction-file argument is optional. It writes one integer class per test row. `k` must be positive and no larger than the training set. CUDA block size must be positive and fit the GPU. Device 0 means the first GPU visible inside the assigned Slurm task, not necessarily physical GPU 0 on the host.

`verify_accuracy.sh` first runs synthetic cases for neighbor ties, class-vote ties, several `k` values, invalid arguments, and partially occupied blocks. It then compares full prediction files against serial for each dataset and block size. Matching aggregate accuracy alone is insufficient.

The serial KNN scan and insertion algorithm remain the Assignment 1 reference. Changes add portable standard headers, input checks, optional prediction output, fractional timing, and release of flattened input buffers. Accuracy is counted directly from predictions so test sets may omit a training class without indexing outside a confusion matrix.

## Baseline measurements after correctness passes

Use the same verified resource options and submit the job with `benchmark` instead of `verify`.

```bash
export DATASETS='small medium large' BLOCK_SIZES='64 128 256' REPS=3 K=3
# Use the confirmed partition/account/GPU options for this cluster.
sbatch --partition=REPLACE_WITH_GPU_PARTITION --gres=gpu:1 \
  --output=slurm/logs/baseline-%j.log \
  slurm/baseline_job.sh benchmark
```

The benchmark records three serial runs per dataset and three CUDA runs per dataset/block size. Every measured prediction file is compared against the serial reference. Each invocation creates a unique directory under `results/baseline-*` containing raw CSV measurements, per-run logs and predictions, metadata, and a summary of average/minimum/maximum time and speedup. A failed run stops the script and preserves diagnostic artifacts. A summary is written only after the requested sweep completes.

Use `REPS=1 DATASETS=small` for a short trial, not final report evidence. The job defaults to 30 minutes and 4 GB of host memory. Adjust those requests after inspecting the system and measuring the first job. Record timeouts rather than treating incomplete runs as successful measurements.

Each measured run launches a fresh process without an explicit warm-up. End-to-end classification time includes flattening matrices, allocation, transfers, GPU work, and workspace cleanup. It excludes ARFF parsing, validation, device discovery, accuracy calculation, and prediction-file output. The returned prediction buffer remains alive at the end of both CPU and GPU timers. GPU kernel time is measured separately with synchronized CUDA events. The headline speedup divides serial classification time by GPU end-to-end classification time.

The default CPU flags use `-O2 -ffp-contract=off`. CUDA uses `-O2 --fmad=false` to retain separate multiply/add rounding in the distance loop. Neither build uses fast math. These settings support reproducibility but still require prediction checks on each system. Old Assignment 1 timings used different build settings and must not be mixed into these speedups.

Preserve the Slurm log with the result directory. It records hardware, compiler versions, and the actual build commands. To return the artifacts after jobs finish, create a uniquely named archive inside Assignment 2 and download it using your normal file-transfer workflow.

```bash
tar -czf "results/baseline-artifacts-$(hostname)-$(date +%Y%m%dT%H%M%S).tar.gz" \
  results/cluster-*.txt results/baseline-*/ slurm/logs/
```

Only run that archive command after the inspection and benchmark outputs exist. Logs, generated results, and binaries are ignored by Git. Send the archive for review instead of assuming a normal Git push will include them.

## Local checks and next milestone

The serial code and synthetic tests can run without CUDA.

```bash
make serial
python3 tests/check_predictions.py
python3 tests/test_benchmark.py
```

`make cpu` retains optional pthreads, OpenMP, and MPI targets for later CPU comparisons. Those inherited programs need their respective compiler support and are not exercised by the baseline GPU job.

Personal explanations are in [Explanations](../Explanations/). After successful cluster verification, the next implementation steps are shared-memory reuse and MPI+CUDA. No speedup claim is made until actual GPU results are available.

See [VALIDATION.md](VALIDATION.md) for completed local checks and the remaining cluster checks.

## Verified Maple workflow, October 22, 2026

Maple uses partition `gpu`, account `egr`, QOS `normal`, and `--gres=gpu:1`.
The user association permits aggregate resources of 2 CPUs, 1 GPU, and 16 GB.
The partition maximum wall time is 30 days. No separate job-count or submission-count limit was reported for this association. Run these jobs sequentially.
The allocated GPU is a GTX 1080 Ti with 11264 MiB. Modules are `gcc/11` and `cuda/12.3`.

These exact commands were used from this directory. The first job was 1470 and the full verification job was 1471.

```bash
MODULES='gcc/11 cuda/12.3' DATASETS=small BLOCK_SIZES='64 128 256' K=3 sbatch --parsable --partition=gpu --account=egr --qos=normal --gres=gpu:1 --time=00:05:00 --output=slurm/logs/baseline-%j.log slurm/baseline_job.sh verify
MODULES='gcc/11 cuda/12.3' DATASETS='small medium large' BLOCK_SIZES='64 128 256' K=3 sbatch --parsable --partition=gpu --account=egr --qos=normal --gres=gpu:1 --output=slurm/logs/baseline-%j.log slurm/baseline_job.sh verify
MODULES='gcc/11 cuda/12.3' DATASETS='small medium large' BLOCK_SIZES='64 128 256' REPS=3 K=3 sbatch --parsable --partition=gpu --account=egr --qos=normal --gres=gpu:1 --output=slurm/logs/baseline-%j.log slurm/baseline_job.sh benchmark
squeue -u "$USER"
sacct -j 1470,1471 --format=JobID,State,ExitCode,Elapsed,AllocTRES%80
```

Submit the benchmark only after full verification passes. The batch script executes `make -B all` inside the allocation. Its Makefile uses `g++ -I libarff -O2 -std=c++11 -ffp-contract=off` and `nvcc -I libarff -O2 -std=c++11 --fmad=false` with the corresponding source and parser files. Full commands are preserved in each job log.

Verification retains dataset predictions in `results/verify.*` and synthetic inputs, commands, outputs, and predictions in `results/correctness-*`. Job logs include CPU/GPU information and compiler versions. Each job writes its source diff and uses an Assignment 2 temporary directory.

## Transfer and inspect the cleaned Maple evidence

The tracked archive `../maple-baseline-evidence-20261022.tar.gz` contains the raw evidence for jobs 1470–1472. Its adjacent `.sha256` file verifies the download. The original working results remain ignored by Git. This archive is a deliberately published snapshot.

After pulling the repository, run from `Assignment2_folder`.

```bash
sha256sum -c maple-baseline-evidence-20261022.tar.gz.sha256
mkdir -p HPDS_assignment2/results
tar -xzf maple-baseline-evidence-20261022.tar.gz -C HPDS_assignment2/results
```

Open `HPDS_assignment2/results/maple-baseline-evidence-20261022/README.md`. The bundle groups CSVs, logs, and predictions under `measurements/`, with separate `verification/`, `slurm/`, `environment/`, `provenance/`, `diagnostics/`, and `source/` directories. `manifest.csv` records original paths and SHA-256 digests. The included summary preserves the original Maple paths, which the manifest maps to the bundle layout.

The packaging script `tools/package_maple_results.py` copies existing evidence and verifies all 36 measured predictions and archived file digests. It omits binaries, Python caches, and empty compiler temporary directories. It refuses to overwrite an existing bundle. It does not rerun the workload or change measurements.

Slurm already writes benchmark output directly into a unique results directory. The additional top-level working files came from discovery and separate verification jobs. Use the grouped bundle for transfer and analysis; retain the original evidence for traceability.

## Verified Athena workflow, October 22, 2026

Athena settings were discovered separately and do not reuse Maple or Hickory values. Slurm is 23.11.10. GPU partitions are `gpu-v100`, `gpu-a100`, `gpu-h100` and the combined `gpu`. A GPU is requested with `--gres=gpu:1`. QOS `normal` is the only QOS for this user, and no `--account` option was needed. The only limits found for the QOS are per-user aggregates of 13 GPUs, 200 running jobs and 1200 CPUs. Partitions report no wall-time limit.

The login node has no `nvcc` and no GPU. CUDA is a module that appears on GPU nodes, with versions `cuda/12.3`, `cuda/12.9` and `cuda/13.2`. The baseline uses `cuda/12.3` with the node's default `g++` 11.4.1. The sweep ran on an NVIDIA A100-SXM4-80GB (compute capability 8.0, driver 595.91.07). An H100 80GB node (athena531) also exists. A V100 discovery job stayed pending and was cancelled, so no V100 information was collected. GPU counts for later work are 8 A100 GPUs (athena521 and athena522, 4 each), 20 H100 GPUs (athena531 has 8, athena532 to athena534 have 4 each) and V100 nodes with 2 GPUs each (athena512 to athena514 are listed by sinfo). Multi-node runs therefore look possible, but they have not been tried. Only one GPU was requested for this baseline.

All Athena output goes to `results/athena/` and `slurm/logs/athena/`. The scripts accept `RESULTS_ROOT` (default `results`, relative to this directory) and `RUN_ID` (default empty, or `job<ID>` when `RESULTS_ROOT` is set). Maple defaults are unchanged. The submit script builds with an explicit `-gencode arch=compute_80,code=sm_80` so that no PTX compilation occurs inside the first timed kernel launch. The other flags are unchanged (`-O2 -std=c++11 --fmad=false`).

Run from this directory. Each command below is run only after the previous job has completed with exit code 0:0.

```bash
bash slurm/athena/submit_athena.sh small      # build and small-dataset verification
bash slurm/athena/submit_athena.sh verify     # all datasets, synthetic cases, blocks 64 128 256
bash slurm/athena/submit_athena.sh benchmark  # 9 serial and 27 CUDA measurements
bash slurm/athena/monitor_athena.sh JOBID [JOBID ...]
```

Each script prints the job ID and appends the exact `sbatch` arguments to `results/athena/submissions.txt`. The underlying command is the following.

```bash
MODULES='cuda/12.3' RESULTS_ROOT=results/athena K=3 BLOCK_SIZES='64 128 256' DATASETS='small medium large' REPS=3 \
NVCCFLAGS='-O2 -std=c++11 --fmad=false -gencode arch=compute_80,code=sm_80' \
sbatch --parsable --partition=gpu-a100 --qos=normal --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --job-name=athena-benchmark --time=00:30:00 --output=slurm/logs/athena/baseline-%j.log slurm/baseline_job.sh benchmark
squeue -u "$USER"
sacct -X -j JOBID --format=JobID,State,ExitCode,Elapsed,AllocTRES%60
```

After the benchmark, `python3 -I tools/audit_athena_results.py results/athena/baseline-DIRECTORY JOB1 JOB2 JOB3` recomputes counts, prediction comparisons, log agreement, summaries and Slurm states. See [ATHENA_BASELINE_RESULTS.md](../ATHENA_BASELINE_RESULTS.md) for results. The evidence archive is `../athena-baseline-evidence-20261022.tar.gz`, built by `tools/package_athena_results.py`.
