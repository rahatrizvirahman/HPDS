# Remaining experiments and report status

Reviewed after pulling commit `7ef8264`. This is a review of saved evidence, not a live query of either cluster. The report is based on completed runs through October 23, 2026.

## What is complete

- Maple global-memory baseline, 36 benchmark measurements.
- Athena A100 original global baseline, 36 measurements.
- Athena paired global/shared sweeps on A100 and H100, 63 measurements each.
- Athena serial, pthreads, OpenMP, and CPU MPI comparison, 144 measurements.
- MPI+CUDA full-prediction verification on 1, 2, 3, and 4 H100 GPUs in one node and two V100 nodes. These are correctness runs, not repeated scaling experiments.
- Independent archive and numeric audit of all 342 benchmark measurements, 7,025 manifest entries, and 144 MPI dataset comparisons.

## Completed priority 1. Maple shared memory

Completed on Maple in verification job 1560 and paired benchmark job 1561, both COMPLETED with exit 0:0. All 63 measurements and predictions passed both audits. See MAPLE_SHARED_MEMORY_RESULTS.md and maple-shared-memory-evidence-20261023.tar.gz with its checksum. The new evidence has not been pushed. Do not duplicate this completed sweep. The commands below are retained for reproducibility.

Run from `Assignment2_folder/HPDS_assignment2` on Maple. Preserve local changes before pulling. Do not compile concurrent jobs in the same checkout because they share executable paths.

```bash
git status --short
git pull --ff-only
squeue -u "$USER" -o '%i %P %j %T %M %R'
mkdir -p slurm/logs/maple-shared results/maple-shared
unset NVCCFLAGS
MODULES='gcc/11 cuda/12.3' RESULTS_ROOT=results/maple-shared K=3 \
DATASETS='small medium large' BLOCK_SIZES='64 128 256' \
STRATEGIES='global shared' TILE_ROWS=256 \
sbatch --parsable --partition=gpu --account=egr --qos=normal --gres=gpu:1 \
  --time=00:20:00 --output=slurm/logs/maple-shared/verify-%j.log \
  slurm/baseline_job.sh verify
```

Check that verification finishes successfully and its log ends with `ALL PREDICTIONS MATCH SERIAL`. Only then submit the benchmark.

```bash
MODULES='gcc/11 cuda/12.3' RESULTS_ROOT=results/maple-shared K=3 \
DATASETS='small medium large' BLOCK_SIZES='64 128 256' \
STRATEGIES='global shared' TILE_ROWS=256 REPS=3 \
sbatch --parsable --partition=gpu --account=egr --qos=normal --gres=gpu:1 \
  --time=00:30:00 --output=slurm/logs/maple-shared/benchmark-%j.log \
  slurm/baseline_job.sh benchmark
```

Replace the two job-ID placeholders and the directory placeholder below with the returned IDs and actual completed run directory.

```bash
sacct -X -j VERIFY_JOB,BENCHMARK_JOB --format=JobID,State,ExitCode,Elapsed,AllocTRES%80 \
  > results/maple-shared/jobs-accounting.txt
GPU_NAME='1080 Ti' STRATEGIES='global shared' \
python3 -I tools/audit_athena_results.py \
  results/maple-shared/ACTUAL_BASELINE_DIRECTORY VERIFY_JOB BENCHMARK_JOB
```

The audit tool name is historical. Its GPU-name and strategy settings support the Maple sweep. Require `AUDIT PASSED`. Preserve the new results, predictions, job logs, source revision/patch, and GPU/compiler details in a Maple-specific evidence archive with a checksum, following the existing evidence format. Push the archive and summary explicitly; ordinary result folders are ignored by Git. Do not replace the old Maple baseline archive.

## Priority 2. Refresh Athena job 3383610

The saved handoff identifies this as the queued H100 timing job. It would measure 1, 2, and 4 GPUs on all datasets with three repetitions. The MPI archive includes submission history but no raw current queue/accounting snapshot for this pending job. Run the following on Athena and preserve the output even if it is still pending.

```bash
cd /path/to/HPDS/Assignment2_folder/HPDS_assignment2
mkdir -p results/athena-mpi-h100
stamp=$(date -u +%Y%m%dT%H%M%SZ)
{
  date -u
  squeue -j 3383610 -o '%i %P %j %T %M %R'
  sacct -X -j 3383610 --format=JobID,State,ExitCode,Submit,Start,End,Elapsed,AllocTRES%80
  scontrol show job 3383610
} > "results/athena-mpi-h100/queue-evidence-$stamp.txt" 2>&1
```

An empty `squeue` response alone does not establish success. Use accounting and the job log. If the job completed, audit its actual directory before adding measurements to the report.

```bash
GPU_NAME=H100 LAYOUTS='1x1 1x2 1x4' \
python3 -I tools/audit_mpi_cuda_results.py \
  results/athena-mpi-h100/ACTUAL_MPI_BENCHMARK_DIRECTORY 3383610
python3 tools/package_athena_extra_results.py mpi NEW_UNIQUE_TAG
```

The completed sweep should contain 45 measurements, including serial and standalone single-GPU baselines as well as MPI+CUDA configurations. Keep its same-job one-GPU MPI baseline for GPU scaling and efficiency. Do not use standalone CUDA time as the one-rank MPI baseline.

If it remains queued, keep the dated evidence and report that the timed allocation was unavailable at the evidence cutoff. Do not claim that multi-GPU execution was impossible, because successful multi-GPU and two-node correctness runs already exist. Do not state that it was unavailable before the deadline until the actual submission cutoff has arrived.

If a four-GPU allocation cannot be scheduled, a consistently measured 1/2/3-GPU sweep may satisfy the three-count requirement, provided resources and course requirements permit it. That is an optional new experiment, not existing evidence. Avoid overlapping jobs that rebuild the same executable. A change to the queued job or a replacement submission should be coordinated on Athena rather than done blindly here.

## Final report and submission

The Overleaf project is `HPDS_Assignment2_overleaf/`. Its tables and plots are generated from the audited archives, including all individual completed run times in an appendix. Keep hardware and sweep labels intact. The report distinguishes cross-platform CPU comparisons from same-sweep GPU speedups.

Before submission, incorporate any new Maple or MPI benchmark evidence, refresh pending-job wording, and confirm the course's report length/template requirements. The current layout follows Assignment 1's IEEE format because no different template was supplied. A documented resource limit does not automatically waive the rubric's timed GPU scaling requirement. If the timing job remains blocked near submission, explain the specific missing experiment to the professor and include the saved evidence.

Package the final source, parser, datasets, Makefile, scripts, and updated report. Keep personal explanations and historical Assignment 1 artifacts outside the submission. No Excel report is required. No cluster job was submitted or cancelled during this local review.
