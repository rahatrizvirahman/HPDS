# Report handoff

State as of October 23, 2026. Numbers below come from the audited CSV files in this repository. The assignment is due October 25, 2026. Items marked **outstanding** have no data yet and must not be reported as results.

## Status against the assignment requirements

| Requirement | Status | Evidence |
|---|---|---|
| Single-GPU CUDA program, correct on all datasets | Done on Maple (global kernel), Athena A100 and Athena H100 | `MAPLE_BASELINE_RESULTS.md`, `ATHENA_BASELINE_RESULTS.md`, `ATHENA_H100_SINGLE_GPU_RESULTS.md` |
| At least two distinct kernel strategies | Implemented (global memory, shared-memory tiling) | `cuda.cu`, `knn_cuda.cuh`, `Explanations/02-shared-memory.md` |
| Maple results, all datasets, every submitted strategy | **Outstanding for the shared kernel.** Maple has the global kernel only | Commands below |
| Athena results, three runs and averages, every CUDA version, all datasets | Done for global and shared on A100 and on H100 | `ATHENA_SHARED_MEMORY_RESULTS.md`, `ATHENA_H100_SINGLE_GPU_RESULTS.md` |
| Kernel configuration experiments (block 64, 128, 256, both strategies, 3 repetitions) | Done on Athena A100 and H100. Maple has the global kernel only | same |
| MPI+CUDA with one GPU per rank, ordered predictions | Implemented and verified (see wording below) | `ATHENA_MPI_CUDA_RESULTS.md` |
| Multi-GPU scaling at three GPU counts, 3 datasets, 3 repetitions | **Outstanding. Not measured.** Job 3383610 was queued | `ATHENA_MPI_CUDA_RESULTS.md` |
| Multi-GPU efficiency | **Outstanding** (needs the scaling run) | none |
| Multi-node evidence | Correctness verified on 2 V100 nodes. No timed multi-node run. | `ATHENA_MPI_CUDA_RESULTS.md` |
| CPU comparison (serial, pthreads, OpenMP, MPI) and speedup | Done on Athena (one CPU node type). Not done on Maple | `ATHENA_CPU_RESULTS.md` |
| Overleaf report | **Outstanding** (you are writing it) | this folder |
| Submission archive (source, scripts, datasets, Makefile, report) | **Outstanding** | none |

## Suggested wording for the multi-GPU limitation

Use this and not "we did not get a multi-GPU allocation", which would be inaccurate, because multi-GPU allocations were obtained for verification.

> MPI+CUDA was implemented with one rank per GPU and verified against the serial predictions on 1 to 4 H100 GPUs in one node, and on two nodes with one V100 GPU each. We could not obtain an allocation for the timed multi-GPU scaling experiment on Athena before the deadline, because no node with four free GPUs became available (the benchmark job remained queued), so we report no multi-GPU speedup or efficiency. Launches involving one H100 node, athena534, failed inside Slurm and were avoided.

The A100 nodes were fully allocated by other users, so an A100 multi-GPU job also never started. If job 3383610 completes after you pull, follow "If job 3383610 runs later" in `ATHENA_MPI_CUDA_RESULTS.md` before using its numbers.

## Hardware and what ran where

Never pool numbers across rows of this table.

| Platform | GPU | Host CPU | Used for |
|---|---|---|---|
| Maple | GTX 1080 Ti, 11264 MiB | Xeon E5-2690 v4 | Baseline (global) sweep, 36 runs |
| Athena athena521 | A100-SXM4-80GB | AMD EPYC 7763 | Global sweep (36 runs), global and shared sweep (63 runs) |
| Athena athena533 | H100 80GB HBM3 | Xeon Platinum 8462Y+ | Global and shared sweep (63 runs), 1 and 2 rank MPI verification |
| Athena athena531 | H100 80GB HBM3 | Xeon Platinum 8480CL | 1 to 4 rank MPI verification |
| Athena athena513, athena514 | V100 | not recorded here | Two-node MPI verification only |
| Athena athena201 | none | AMD EPYC 7742, 16 CPUs allocated | CPU comparison, 144 runs |

Speedup always means the mean serial classification time measured in the same job divided by the mean GPU end-to-end classification time. Serial times differ between jobs because the host CPUs differ, so GPU speedups from different platforms should not be compared. Absolute GPU times can be compared if labeled. Use the Maple serial time for Maple speedups and the Athena serial time for Athena speedups.

## Timing definitions (state these in the methodology)

End-to-end classification time starts after ARFF parsing and includes matrix flattening, device allocation, host-to-device copies, the kernel, the copy back, and cleanup. It excludes parsing, validation, device discovery, accuracy calculation and prediction-file output. Kernel time uses synchronized CUDA events and is reported separately. Each repetition is a fresh process with no warm-up. The CPU programs use the same boundary. The MPI programs report the maximum over ranks. Block sizes were 64, 128 and 256 threads, k=3, and the shared kernel used a fixed 256-row tile.

## Where the data is

- `report_data/` holds audited CSV copies for tables and figures (raw rows, summaries and kernel summaries) with provenance and SHA-256 digests.
- Archives, each with a `.sha256` file and an internal `manifest.csv`, are in this folder.
  - `maple-baseline-evidence-20261022.tar.gz` (Maple baseline)
  - `athena-baseline-evidence-20261022.tar.gz` (A100, global kernel)
  - `athena-shared-memory-evidence-20261022.tar.gz` (A100, global and shared kernels)
  - `athena-h100-single-gpu-evidence-20261023.tar.gz` (H100 single GPU)
  - `athena-cpu-evidence-20261023.tar.gz` (CPU comparison)
  - `athena-mpi-cuda-evidence-20261023.tar.gz` (MPI+CUDA verification, failed jobs and launch diagnostics)
- To unpack on a Mac, from this folder run `shasum -a 256 -c NAME.tar.gz.sha256` and then `tar -xzf NAME.tar.gz -C SOME_EMPTY_DIRECTORY`.

## What the data supports

- Every GPU and CPU configuration listed above produced predictions identical to serial for all test rows, and accuracies are 85.00000000%, 27.56756783% and 99.47613525% for small, medium and large. Equal accuracy alone is not the evidence, because the full prediction files were compared.
- On Athena the shared kernel was faster than the global kernel at the same block size on every dataset on both GPUs. The ratio is 1.10 to 1.28 on the A100 and 1.02 to 1.17 on the H100. Block 64 and 128 are close for the shared kernel, and the lowest mean is at block 128 on the H100.
- The small dataset (160 queries) is slower on the GPU than serial in every configuration. Fixed allocation, transfer and launch costs are a plausible explanation, and no profiler measurement isolates them.
- The explanation that shared memory helps because threads of a block reuse training rows is plausible and was not measured. No profiler was used.
- The CPU programs scale well to about 8 workers on medium and large. Pthreads and OpenMP gained little or nothing from 8 to 16 workers on the 16 CPUs allocated, while MPI kept improving. Variation between repetitions is large, so show the minimum and maximum.

## Things not to claim, and caveats

- No timed multi-GPU or multi-node scaling, speedup or efficiency exists.
- The global kernel measured 2% to 4% slower on large in the A100 shared-memory job than in the A100 baseline job (same node and flags, run after the helper refactor). The cause was not isolated, so treat the two sweeps as separate.
- The Athena builds name the GPU architecture (`sm_80` or `sm_90`) and Maple's did not, so build flags differ between systems. The arithmetic flags are the same (`--fmad=false`, no fast math).
- The A100 baseline and shared-memory archives were produced before the kernels moved into `knn_cuda.cuh`. The kernel code is the same, and each archive contains the source used.
- Assignment 1 timings used different flags and integer-millisecond timers, and are not comparable.

## Outstanding work that needs you

### Maple: shared-memory kernel (the main gap in the rubric)

Run from `Assignment2_folder/HPDS_assignment2` on Maple after `git pull --ff-only`. Do not set `NVCCFLAGS`, so the build matches the existing Maple baseline. Run one job at a time. The verification job must finish with `ALL PREDICTIONS MATCH SERIAL` before you submit the benchmark.

```bash
MODULES='gcc/11 cuda/12.3' RESULTS_ROOT=results/maple-shared K=3 DATASETS='small medium large' BLOCK_SIZES='64 128 256' \
STRATEGIES='global shared' TILE_ROWS=256 sbatch --parsable --partition=gpu --account=egr --qos=normal --gres=gpu:1 \
--time=00:20:00 --output=slurm/logs/maple-shared-verify-%j.log slurm/baseline_job.sh verify

MODULES='gcc/11 cuda/12.3' RESULTS_ROOT=results/maple-shared K=3 DATASETS='small medium large' BLOCK_SIZES='64 128 256' \
STRATEGIES='global shared' TILE_ROWS=256 REPS=3 sbatch --parsable --partition=gpu --account=egr --qos=normal --gres=gpu:1 \
--time=00:30:00 --output=slurm/logs/maple-shared-bench-%j.log slurm/baseline_job.sh benchmark

squeue -u "$USER"
sacct -j VERIFYJOB,BENCHJOB --format=JobID,State,ExitCode,Elapsed
GPU_NAME='1080 Ti' STRATEGIES='global shared' python3 -I tools/audit_athena_results.py results/maple-shared/baseline-DIRECTORY VERIFYJOB BENCHJOB
tar -czf maple-shared-evidence-YYYYMMDD.tar.gz results/maple-shared slurm/logs/maple-shared-*.log
```

The audit expects 63 rows and prints `AUDIT PASSED` when everything agrees. Maple has a single GPU, so no multi-GPU run is possible there.

### Other

- The queued Athena scaling job (3383610), if it completes (see above).
- The Overleaf report and the final submission archive (source, scripts, datasets, Makefile and report). The personal notes in `Explanations/` and the archived Assignment 1 material should stay out of the submission.
