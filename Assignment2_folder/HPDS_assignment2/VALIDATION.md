# Baseline validation status

## Checked locally on October 22, 2026

- Built the serial reference with Apple Clang and the installed macOS 15.4 SDK. The default macOS 27 SDK failed to link in this environment, so the older SDK was selected only for the local test command. No macOS SDK path is embedded in the Makefile.
- Compared every serial prediction with the submitted Assignment 1 algorithm on small, medium, and large datasets at k equal to 3. All predictions matched. Both local comparison builds used the same optimization and arithmetic flags.
- Passed synthetic serial checks for equal-distance neighbor ordering, class-vote ties, k equal to 1, 2, and the full training count, invalid k values, and a test set omitting a training class.
- Passed benchmark collection tests using synthetic process output. Invalid measurements and prediction mismatches are rejected, completed summaries calculate the expected speedup, and partial failures retain raw evidence.
- Checked Bash syntax and Python compilation for the new workflow scripts.
- Verified that all supplied dataset and parser files remain identical to Assignment 1.
- Passed `git diff --check`.

The unchanged parser emits a compiler deprecation warning for `vsprintf` on this Mac. This did not prevent the local serial build with the compatible SDK.

## Still required on the clusters

This machine has no `nvcc` or CUDA execution environment. The CUDA source has not yet been compiled or run. Local serial and script tests are not evidence of GPU correctness or performance.

Run the discovery workflow in `README.md` on Maple and Athena first. Confirm their modules and Slurm GPU requests, then submit the verification job. It rebuilds the CUDA program, runs the synthetic tests on the real GPU, and compares all supplied-dataset predictions at block sizes 64, 128, and 256.

Record each system's successful build and verification log before collecting report measurements. Shared-memory and MPI+CUDA validation followed in later milestones (see the Athena sections below).

## Maple validation completed October 22, 2026

The earlier unvalidated status above describes the local authoring environment. Maple jobs 1470, 1471, and 1472 all completed with exit code 0:0. CUDA built successfully with CUDA 12.3 and GCC 11.4 on a GTX 1080 Ti. Synthetic checks and exact predictions passed for all datasets at k=3 and blocks 64, 128, and 256. All 36 benchmark prediction files matched serial. See [Maple baseline results](../MAPLE_BASELINE_RESULTS.md) for measurements, settings, changes, and artifact paths. Athena, shared memory, and MPI+CUDA remain unvalidated.

## Athena validation completed October 22, 2026

Athena jobs 3382502 (small), 3382504 (all datasets) and 3382505 (benchmark) completed with exit code 0:0 on an A100-SXM4-80GB with CUDA 12.3. Synthetic checks and complete prediction files matched serial for every dataset at k=3 and blocks 64, 128 and 256, and all 36 benchmark predictions matched. See [Athena baseline results](../ATHENA_BASELINE_RESULTS.md). Shared memory and MPI+CUDA remain unvalidated.

## Athena shared-memory validation completed October 22, 2026

Athena jobs 3382580, 3382581 and 3382591 completed 0:0. The shared-memory strategy and the refactored global kernel matched serial on every dataset at k=3 and blocks 64, 128 and 256, with synthetic partial-tile and partial-block cases passing. All 63 benchmark predictions matched. See [Athena shared-memory results](../ATHENA_SHARED_MEMORY_RESULTS.md). Maple validation of the shared kernel and MPI+CUDA remain outstanding.

## Athena H100, CPU and MPI+CUDA validation, October 23, 2026

H100 single-GPU jobs 3382994 and 3382996 completed 0:0, with all 63 benchmark predictions matching serial. CPU jobs 3383689, 3383691 and 3383699 completed 0:0, with pthreads, OpenMP and MPI matching serial at 1, 2, 4, 8 and 16 workers and all 144 benchmark predictions matching. MPI+CUDA verification passed on 1 to 4 H100 GPUs in one node (job 3382997), on 1 and 2 H100 GPUs (job 3382992) and on 2 V100 nodes (job 3382990). The timed MPI+CUDA scaling benchmark (job 3383610) was still queued and has no results. See [ATHENA_H100_SINGLE_GPU_RESULTS.md](../ATHENA_H100_SINGLE_GPU_RESULTS.md), [ATHENA_CPU_RESULTS.md](../ATHENA_CPU_RESULTS.md) and [ATHENA_MPI_CUDA_RESULTS.md](../ATHENA_MPI_CUDA_RESULTS.md).
