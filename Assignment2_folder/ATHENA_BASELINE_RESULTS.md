# Athena single-GPU baseline results

Completed October 22, 2026. This completes the Athena global-memory baseline milestone only. Shared memory, MPI+CUDA, multi-GPU scaling, CPU parallel comparisons and the Overleaf report remain later work. Nothing was pushed or uploaded. Athena speedups use Athena serial times only. No Maple or Hickory timing appears in any Athena calculation.

## Environment and allocation

All measured work ran on node athena521 in partition `gpu-a100`. The GPU was an NVIDIA A100-SXM4-80GB (81920 MiB, compute capability 8.0, driver 595.91.07). The CPU was a dual-socket AMD EPYC 7763 (128 cores in total). Slurm is 23.11.10. CUDA 12.3.107 came from `module load cuda/12.3`, and the host compiler was the node's default g++ 11.4.1 (20231218). The login node has g++ 11.5.0, no `nvcc` and no GPU, so nothing was built or run there.

Every job requested `--partition=gpu-a100 --qos=normal --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=1`, with the 4 GB memory default from the job script. No account option was needed, because the default association (account `rahmanr12`, QOS `normal`) is accepted. The only limits found are per-user aggregates for QOS `normal` (13 GPUs, 200 running jobs, 1200 CPUs, 10 TB memory). Partitions report no wall-time limit. Jobs ran sequentially, with limits of 10, 20 and 30 minutes.

Other GPU models exist. athena531 has H100 80GB GPUs, and a one-GPU discovery job there (3382472) completed. A V100 discovery job (3382470) stayed pending behind higher priority work and was cancelled, so no V100 details were collected. The sweep used the A100 only, so no measurements from different hardware were combined.

For later work, the nodes expose these GPU counts.

| Partition | Nodes and GPUs per node | Total GPUs |
|---|---|---|
| gpu-a100 | athena521, athena522 with 4 each | 8 |
| gpu-h100 | athena531 with 8, athena532 to athena534 with 4 each | 20 |
| gpu-v100 | athena512 to athena514 with 2 each (as listed by sinfo) | 6 to 8 |

Multi-node jobs look possible, but none were attempted.

## Build

Each job ran `make -B all` inside its allocation. The log preserves the expanded commands.

```
g++ -I libarff -O2 -std=c++11 -ffp-contract=off serial.cpp libarff/*.cpp -o serial
nvcc -I libarff -O2 -std=c++11 --fmad=false -gencode arch=compute_80,code=sm_80 cuda.cu libarff/*.cpp -o cuda
```

The Maple build did not name a GPU architecture. For Athena the submit script adds `-gencode arch=compute_80,code=sm_80` through `NVCCFLAGS`, so the kernel is compiled for the A100 and not recompiled from PTX at its first launch. This was a precaution to keep one-time compilation out of the timed regions. Its effect was not measured, because no run without the flag was made. The arithmetic flags (`--fmad=false`, no fast math) are unchanged, and every prediction still matched serial.

The tested revision is `d578e1ddb97e5290ad06bc3ffd1128e65d9a66bd` plus uncommitted workflow changes. Each job saved `git status` and `git diff` (`job-<ID>.status.txt` and `job-<ID>.patch`). The new files under `slurm/athena/` and `tools/` are untracked, so the archive carries complete source snapshots.

## Commands and jobs

```bash
bash slurm/athena/submit_athena.sh small
bash slurm/athena/submit_athena.sh verify
bash slurm/athena/submit_athena.sh benchmark
bash slurm/athena/monitor_athena.sh JOBID
```

The script's exact `sbatch` arguments are in `results/athena/submissions.txt` and in the README. Environment values were `MODULES=cuda/12.3`, `RESULTS_ROOT=results/athena`, `K=3`, `BLOCK_SIZES='64 128 256'`, and `DATASETS`/`REPS` as stated below.

| Job | Purpose | Final state | Exit | Elapsed |
|---|---|---|---|---|
| 3382470 | V100 discovery (never started) | CANCELLED | 0:0 | 0 s |
| 3382471 | A100 discovery | COMPLETED | 0:0 | 4 s |
| 3382472 | H100 discovery | COMPLETED | 0:0 | 7 s |
| 3382502 | Build and small-dataset verification | COMPLETED | 0:0 | 23 s |
| 3382504 | All-dataset and synthetic verification | COMPLETED | 0:0 | 27 s |
| 3382505 | Benchmark, 9 serial and 27 CUDA runs | COMPLETED | 0:0 | 41 s |

Batch and `srun` steps of jobs 3382502, 3382504 and 3382505 also finished COMPLETED with 0:0. Full accounting is in `results/athena/jobs-accounting.txt`.

## Correctness

Synthetic cases passed for equal-distance neighbor ordering, class-vote ties, k=1, 2 and 4, invalid k values (0, -1, 5, abc, 3x) and partial blocks at 1, 64, 128 and 256 threads. In job 3382504 the complete prediction files for small, medium and large matched serial at k=3 for blocks 64, 128 and 256 (nine comparisons). Accuracies were 85.00000000% (small), 27.56756783% (medium) and 99.47613525% (large) for serial and every CUDA configuration, matching the Maple values. No check was weakened and no code change was needed to make Athena pass.

## Benchmark

The benchmark used K=3, all three datasets, blocks 64, 128 and 256, and three repetitions (job 3382505). It produced 36 rows. Each of the twelve dataset/strategy/block groups has three repetitions. Each run is a fresh process with no explicit warm-up.

End-to-end classification time includes flattening, allocation, transfers, GPU execution, prediction retrieval and cleanup, and excludes ARFF parsing, validation, device discovery, accuracy calculation and prediction output. Kernel time uses synchronized CUDA events and is reported separately. Speedup is mean Athena serial classification time divided by mean A100 end-to-end classification time. Times are in milliseconds.

| Dataset | Strategy | Block | Runs | Mean end-to-end | Min | Max | Mean kernel | Speedup |
|---|---|---|---|---|---|---|---|---|
| small | serial | 0 | 3 | 1.380346 | 1.373387 | 1.385449 | n/a | 1.0000 |
| small | global | 64 | 3 | 1.581222 | 1.545100 | 1.624330 | 1.101141 | 0.8730 |
| small | global | 128 | 3 | 1.592222 | 1.560459 | 1.634308 | 1.099435 | 0.8669 |
| small | global | 256 | 3 | 1.628176 | 1.621624 | 1.633927 | 1.137664 | 0.8478 |
| medium | serial | 0 | 3 | 103.500258 | 103.390367 | 103.651740 | n/a | 1.0000 |
| medium | global | 64 | 3 | 15.436224 | 15.415495 | 15.468024 | 13.295616 | 6.7050 |
| medium | global | 128 | 3 | 16.961787 | 16.060129 | 18.731791 | 13.901824 | 6.1020 |
| medium | global | 256 | 3 | 16.443321 | 16.082782 | 17.006723 | 14.187179 | 6.2944 |
| large | serial | 0 | 3 | 1987.591580 | 1987.158067 | 1987.941753 | n/a | 1.0000 |
| large | global | 64 | 3 | 62.408911 | 62.322105 | 62.539804 | 54.661174 | 31.8479 |
| large | global | 128 | 3 | 64.278447 | 63.080724 | 65.167685 | 56.892320 | 30.9216 |
| large | global | 256 | 3 | 68.249502 | 68.165611 | 68.370247 | 60.568769 | 29.1224 |

Block 64 had the lowest mean on every dataset. On small, blocks 64 and 128 differ by only 0.011 ms and their repetition ranges overlap, so the ordering there is weak evidence. The small dataset ran slower on the GPU than serial (speedup below 1). Small has only 160 queries, and kernel time is about 70% of its GPU end-to-end time, so fixed allocation, transfer and launch costs weigh heavily. These are plausible explanations, and no separate measurement isolates them. Kernel time is 82% to 89% of end-to-end time on medium and large (see `kernel_fraction_of_end_to_end` in `kernel-summary.csv`). Medium block 128 shows the widest spread (16.06 to 18.73 ms), so its mean should be read with that variation in mind. Three repetitions support a baseline comparison, not a broad statistical claim. The node was shared with other jobs (state MIXED), which may add noise to CPU timings in particular.

## Independent audit

`tools/audit_athena_results.py` was written separately from the benchmark collector and recomputed everything from the raw files (`results/athena/audit.txt`). It confirmed the following.

- 36 rows in `raw.csv`, with 12 groups of three repetitions (9 serial rows, 27 CUDA rows).
- All 36 prediction files are byte-identical to their dataset's first serial prediction file.
- Every CSV end-to-end time, kernel time, accuracy and strategy equals the single `RESULT` record in its individual process log.
- Means, minima, maxima, run counts and speedups in `summary.csv` match recomputation. Kernel means were recomputed into `kernel-summary.csv`.
- Jobs 3382502, 3382504 and 3382505 are COMPLETED with exit code 0:0 in Slurm accounting.

## Code and workflow changes

No algorithm, kernel, parser or dataset change was required. Changes were limited to workflow scripts.

- `RESULTS_ROOT` (default `results`) and `RUN_ID` were added to `slurm/baseline_job.sh`, `verify_accuracy.sh`, `benchmark_baseline.py`, `tests/check_predictions.py` and `tests/test_benchmark.py`. They cover synthetic tests, prediction verification, benchmarks, patches and temporary files. This was needed to keep Athena evidence apart from Maple evidence. Defaults are unchanged.
- `slurm/baseline_job.sh` also saves `git status --short`, because `git diff` alone omits untracked files.
- New Athena files are `slurm/athena/submit_athena.sh`, `slurm/athena/monitor_athena.sh`, `slurm/athena/discover_gpu_node.sh`, `tools/audit_athena_results.py` and `tools/package_athena_results.py`. The Maple packager is untouched.
- The README, VALIDATION.md and the personal explanations record the Athena findings. Maple conclusions are unchanged.

## Limitations

- Only one GPU model (A100) and one node (athena521) were measured. H100 and V100 results do not exist.
- The build used an explicit sm_80 target that the Maple build did not, so build flags differ between systems.
- Block size and speedup conclusions come from three repetitions per configuration.
- The 29 to 32 times speedups on large compare against a serial CPU baseline on a shared AMD EPYC node. They are not comparable to Maple speedups, which used a different CPU.
- Multi-node and multi-GPU behavior was not tested.
- This is the single-GPU global-memory baseline, not the completed assignment.

## Artifact paths

Paths are relative to Assignment2_folder. The original working evidence stays under `HPDS_assignment2/results/athena/` and `HPDS_assignment2/slurm/logs/athena/`, which Git ignores.

- `HPDS_assignment2/results/athena/cluster-athena-inspect.txt`, `discovery-details.txt`, `discovery-partitions-modules.txt`, `discover-gpu-*.log`, `a100-nodes-state.txt` and `slurm-settings.txt` hold discovery output.
- `HPDS_assignment2/results/athena/submissions.txt` and `jobs-accounting.txt` hold submissions and Slurm accounting.
- `HPDS_assignment2/slurm/logs/athena/baseline-3382502.log`, `baseline-3382504.log` and `baseline-3382505.log` hold build, hardware and execution logs.
- `HPDS_assignment2/results/athena/verify.job*`, `correctness-job*` hold dataset and synthetic verification evidence.
- `HPDS_assignment2/results/athena/baseline-20261022T220759Z-job3382505-biaftwm5/` holds `raw.csv`, `summary.csv`, `kernel-summary.csv`, `metadata.txt`, and 36 process logs and prediction files.
- `HPDS_assignment2/results/athena/audit.txt`, `job-*.patch`, `job-*.status.txt` and `final-changes.patch` hold the audit and provenance.
- `athena-baseline-evidence-20261022.tar.gz` and its `.sha256` file hold the evidence archive, with `manifest.csv` mapping archived files to original paths and SHA-256 digests.
