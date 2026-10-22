# Athena shared-memory strategy results

Completed October 22, 2026. This adds a second single-GPU kernel strategy (shared-memory tiling) and measures it against the global-memory kernel on Athena. MPI+CUDA, multi-GPU scaling, Maple validation of the shared kernel, CPU parallel comparisons and the Overleaf report remain later work. Nothing was pushed. The frozen baseline evidence in `athena-baseline-evidence-20261022.tar.gz` was not modified.

## What changed in the code

- `cuda.cu` accepts `--strategy=global|shared` and `--tile=rows` anywhere after the program name. The positional interface and the default (global) are unchanged.
- `knn_shared` keeps one query per thread and the serial arithmetic, scan order and tie rules. A block loads `tile_rows` training rows into shared memory with adjacent threads copying adjacent floats, passes a barrier, lets active query threads scan the tile in ascending row order, and passes a second barrier before the next tile replaces it. Inactive threads of a partial query block skip no barrier. The tile defaults to 256 rows (8192 bytes on small with 8 columns per row, and 12288 bytes on medium and large with 12 columns per row), well under one block's shared memory.
- The insertion, vote and distance logic moved into device helpers used by both kernels, so the tie rules exist once. This edited the baseline kernel, so both strategies were re-verified.
- `STRATEGIES` (default `global`) and `TILE_ROWS` (default 256) were added to `verify_accuracy.sh`, `benchmark_baseline.py` and `tests/check_predictions.py`. A new unit test covers a two-strategy sweep. `tools/audit_athena_results.py` and `tools/package_athena_results.py` take the strategy list or milestone name, with the baseline behavior as the default.
- `slurm/athena/submit_athena_shared.sh` submits the three jobs. Its evidence goes to `results/athena-shared/` and `slurm/logs/athena-shared/`, apart from the baseline evidence.

## Environment and jobs

Same as the baseline. The jobs ran on athena521 in `gpu-a100` with one A100-SXM4-80GB, `cuda/12.3` (nvcc 12.3.107) and g++ 11.4.1. The build was `nvcc -I libarff -O2 -std=c++11 --fmad=false -gencode arch=compute_80,code=sm_80`. Each job used `--qos=normal --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=1`. Jobs ran one after another, and the tested revision is commit `9d6cb4a` plus the uncommitted changes in `results/athena-shared/job-<ID>.patch` and `final-changes.patch`.

| Job | Purpose | Final state | Exit | Elapsed |
|---|---|---|---|---|
| 3382580 | Small verification, both strategies | COMPLETED | 0:0 | 50 s |
| 3382581 | All datasets, both strategies, blocks 64, 128, 256 | COMPLETED | 0:0 | 65 s |
| 3382591 | Benchmark, 9 serial and 54 CUDA runs | COMPLETED | 0:0 | 64 s |

## Correctness

The synthetic cases passed for the global kernel and for the shared kernel with tiles of 1, 3 and 256 rows and blocks of 1, 2, 64, 128 and 256 threads. Tiles of 1 and 3 rows split the four training rows into full and partial tiles, and block size 2 leaves the last query block partial. Invalid options were rejected (unknown strategy, tile 0, a non-numeric tile and a tile larger than shared memory). In job 3382581 all 18 dataset/strategy/block combinations at k=3 matched the complete serial prediction file. All 63 benchmark prediction files matched serial. Accuracies are 85.00000000%, 27.56756783% and 99.47613525% for every configuration.

## Benchmark

Settings were K=3, `DATASETS='small medium large'`, `BLOCK_SIZES='64 128 256'`, `STRATEGIES='global shared'`, `TILE_ROWS=256`, three repetitions, each a fresh process with no warm-up. The tile was held fixed while the block size varied. Timing boundaries are those of the baseline, with kernel time from synchronized CUDA events reported separately. Speedup is mean Athena serial end-to-end time divided by mean GPU end-to-end time. Times are in milliseconds.

| Dataset | Strategy | Block | Mean end-to-end | Min | Max | Mean kernel | Speedup |
|---|---|---|---|---|---|---|---|
| small | serial | 0 | 1.388973 | 1.377581 | 1.395736 | n/a | 1.0000 |
| small | global | 64 | 1.683971 | 1.672396 | 1.694207 | 1.211051 | 0.8248 |
| small | global | 128 | 1.673193 | 1.658720 | 1.680510 | 1.207296 | 0.8301 |
| small | global | 256 | 1.634304 | 1.613245 | 1.666114 | 1.171797 | 0.8499 |
| small | shared | 64 | 1.504937 | 1.487858 | 1.521742 | 1.047211 | 0.9229 |
| small | shared | 128 | 1.494414 | 1.464033 | 1.513567 | 1.033216 | 0.9294 |
| small | shared | 256 | 1.487692 | 1.479964 | 1.494922 | 1.019563 | 0.9336 |
| medium | serial | 0 | 103.567849 | 103.482124 | 103.697380 | n/a | 1.0000 |
| medium | global | 64 | 16.034051 | 16.030461 | 16.036473 | 13.928107 | 6.4592 |
| medium | global | 128 | 16.603425 | 16.518892 | 16.765877 | 14.494379 | 6.2377 |
| medium | global | 256 | 16.737353 | 16.375433 | 17.457918 | 14.680064 | 6.1878 |
| medium | shared | 64 | 13.068511 | 12.974207 | 13.188550 | 11.009024 | 7.9250 |
| medium | shared | 128 | 13.015742 | 12.993534 | 13.050451 | 10.907648 | 7.9571 |
| medium | shared | 256 | 13.736130 | 13.699531 | 13.787276 | 11.609088 | 7.5398 |
| large | serial | 0 | 1987.501639 | 1986.083829 | 1989.098188 | n/a | 1.0000 |
| large | global | 64 | 63.889098 | 63.548581 | 64.241614 | 56.995765 | 31.1086 |
| large | global | 128 | 66.148550 | 64.056436 | 67.240343 | 58.795722 | 30.0460 |
| large | global | 256 | 70.623735 | 70.541511 | 70.719325 | 62.933590 | 28.1421 |
| large | shared | 64 | 52.562509 | 51.982892 | 53.612166 | 45.852661 | 37.8122 |
| large | shared | 128 | 52.589259 | 52.584853 | 52.593450 | 44.960373 | 37.7929 |
| large | shared | 256 | 55.343318 | 54.331579 | 55.912601 | 48.190422 | 35.9122 |

End-to-end ratio of global to shared at the same block size.

| Dataset | Block 64 | Block 128 | Block 256 |
|---|---|---|---|
| small | 1.119 | 1.120 | 1.099 |
| medium | 1.227 | 1.276 | 1.218 |
| large | 1.215 | 1.258 | 1.276 |

The shared kernel had a lower mean than the global kernel in all nine dataset/block pairs. The global kernel took 1.10 to 1.28 times as long end-to-end and 1.15 to 1.33 times as long in kernel time. The best shared configurations are block 256 on small (the range overlaps the other blocks, so this is weak), block 128 or 64 on medium (13.016 against 13.069 ms with overlapping ranges, so effectively a tie), and block 64 or 128 on large (52.563 against 52.589 ms, also a tie). The best global configuration is block 256 on small and block 64 on medium and large. Headline speedups for the best shared configurations are 0.93 (small), 7.96 (medium) and 37.81 (large). The small dataset remains slower than serial on the GPU, with kernel time 69% to 72% of end-to-end time, so fixed costs dominate there.

Why shared memory helped is not measured. The plausible explanation is that all threads of a block read the same training rows, so loading each tile once reduces repeated global-memory reads. Each thread's candidate distances, labels and vote counters still live in global memory, which may limit the gain. No profiler was used, so neither point is established.

In this sweep the global kernel measured 2% to 4% slower on large than in the baseline job (for example 63.889 against 62.409 ms at block 64), while medium and small moved both ways. Both jobs used athena521 and the same flags, but this run also follows the helper refactor, so run-to-run variation and the refactor cannot be separated with the data collected. The two sweeps should not be pooled. Three repetitions support a comparison between strategies measured in the same job, not a broad statistical claim.

## Independent audit

`STRATEGIES='global shared' python3 -I tools/audit_athena_results.py <benchmark directory> 3382580 3382581 3382591` recomputed everything from the raw files (`results/athena-shared/audit.txt`). It found 63 rows in 21 groups of three repetitions, byte-identical predictions for all 63 runs, CSV values equal to the individual process logs, summary means, minima, maxima and speedups matching recomputation, and all three jobs COMPLETED 0:0.

## Limitations

- Only one A100 on one node was measured, and the shared kernel has not been run on Maple.
- The tile size was fixed at 256 rows, so the effect of tile size is unexplored.
- The baseline evidence archive predates the helper refactor. It documents the earlier kernel source it was run with.
- This is not the finished assignment. MPI+CUDA and multi-GPU experiments are still needed.

## Artifact paths

Relative to Assignment2_folder.

- `HPDS_assignment2/results/athena-shared/` holds submissions, accounting, verification and synthetic evidence, `audit.txt`, patches and the benchmark directory `baseline-20261022T233959Z-job3382591-0hypd1j1` (raw.csv, summary.csv, kernel-summary.csv, metadata.txt, 63 logs and predictions).
- `HPDS_assignment2/slurm/logs/athena-shared/job-3382580.log`, `job-3382581.log` and `job-3382591.log` hold build and execution logs.
- `athena-shared-memory-evidence-20261022.tar.gz` and its `.sha256` file hold the evidence archive, with a manifest of original paths and SHA-256 digests.
