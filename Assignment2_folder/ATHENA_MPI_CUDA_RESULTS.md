# Athena MPI+CUDA results

Written October 23, 2026. The MPI+CUDA program is implemented and its predictions were verified against serial on 1 to 4 GPUs of one node and on 2 nodes. **The timed multi-GPU scaling benchmark was not run.** It is queued as job 3383610 and had not started when this was written. This document therefore contains correctness evidence and no scaling or efficiency numbers. Nothing here should be presented as a scaling result.

## What was built

`mpi_cuda.cu` runs one MPI rank per GPU. Every rank keeps the full training set on its GPU, classifies a contiguous range of test queries with the same `knn_global` or `knn_shared` kernel used by the single-GPU program (now in `knn_cuda.cuh`), and rank 0 gathers the predictions with `MPI_Gatherv`. Each rank's GPU is chosen from its node-local rank, and the ranks compare GPU UUIDs and abort if two share one. A failed rank calls `MPI_Abort`, so the other ranks cannot wait indefinitely. Timing starts after a barrier and covers flattening, transfers, the kernel, the gather and cleanup, and the reported time is the maximum over ranks. `make mpi_cuda` builds it and is separate from `make all`. See `Explanations/03-mpi-cuda.md` for the design.

## Environment

- Modules `cuda/12.3` and `mpi/openmpi-4.1.6`. The newer `mpi/openmpi-4.1.8` loads gcc-toolset 13, which CUDA 12.3 does not support.
- Launch with `srun --mpi=pmix --nodes=N --ntasks-per-node=P --gpus-per-task=1 --cpus-per-task=1`.
- H100 jobs used partition `gpu-h100` with `-gencode arch=compute_90,code=sm_90`, and V100 jobs used `gpu-v100` with `sm_70`.
- The 4-GPU run used athena531 (8 H100 80GB HBM3, Intel Xeon Platinum 8480CL host). The 1- and 2-rank run used athena533 (Xeon 8462Y+), and the two-node run used athena513 and athena514 (V100).

## Verification that passed

Each verification job runs the synthetic suite and then compares complete prediction files with serial for every dataset, both kernels and blocks 64, 128 and 256 at k=3 (3 datasets × 2 strategies × 3 blocks per layout). Layouts are written NODESxRANKS_PER_NODE.

| Job | GPU | Layouts | Dataset comparisons | Synthetic cases | State | Exit | Elapsed |
|---|---|---|---|---|---|---|---|
| 3382992 | H100, athena533 | 1x1, 1x2 | 36 of 36 matched | passed | COMPLETED | 0:0 | 2 m 35 s |
| 3382997 | H100, athena531 | 1x1, 1x2, 1x3, 1x4 | 72 of 72 matched | 144 passed | COMPLETED | 0:0 | 9 m 32 s |
| 3382990 | V100, athena513 and athena514 | 1x1, 2x1 (two nodes) | 36 of 36 matched | 72 passed | COMPLETED | 0:0 | 2 m 37 s |

The synthetic suite covers equal-distance neighbors, class-vote ties, k=1, 2 and 4, invalid k and options, uneven and empty rank ranges (3 test queries across 4 ranks leaves one rank empty), partial tiles of 1, 3 and 256 rows, blocks of 1, 2 and 128 threads, and an injected failure of rank 1 that must abort the whole job within 60 seconds. The 1x3 layout also splits the real datasets unevenly (for example 740 medium queries as 246, 247 and 247).

In the 4-rank run each rank reported a different GPU. For the large dataset with the shared kernel and block 128, the per-rank lines were as follows. Times are from a single unrepeated verification run and are not benchmark measurements.

| Rank | Host | PCI bus | Queries | Start |
|---|---|---|---|---|
| 0 | athena531 | 0000:1B:00.0 | 859 | 0 |
| 1 | athena531 | 0000:43:00.0 | 859 | 859 |
| 2 | athena531 | 0000:52:00.0 | 859 | 1718 |
| 3 | athena531 | 0000:61:00.0 | 859 | 2577 |

## Failed and cancelled jobs (preserved)

| Job | What happened | Cause |
|---|---|---|
| 3382607 | A100, 4 GPUs. CANCELLED while pending for over 18 hours, never ran. | Both A100 nodes were fully allocated to other users. Cancelled by me so it could not start later and write into the wrong results. |
| 3382968 | H100, athena533 and athena534. Passed 1x1 and 1x2 synthetic cases, then the first two-node case hung until the 120 s timeout. | Slurm could not launch tasks on athena534 (see below). |
| 3382985 | H100, athena534 only. The first step failed. | `srun: error: Unable to create step ... Invalid Trackable RESource (TRES) specification` on athena534. |
| 3382986, 3382988 | V100, two nodes. Synthetic cases passed, then the dataset stage failed. | A bug in my script. `${MPI_LAUNCH:-mpirun -n {ranks}}` ends at the first closing brace and left a stray `}`, giving `Invalid numeric value "1}" for --cpus-per-task`. Fixed. |
| 3382989 | H100, athena533. `no kernel image is available for execution on the device`. | I ran it at the same time as a V100 job. Both build in the same checkout, and the `sm_70` binary most likely replaced the `sm_90` one. It passed when rerun alone (job 3382992). Jobs must now run one after another. |
| 3382972, 3382982, 3382983, 3382984 | Diagnostic jobs (their own state is COMPLETED, but most launch attempts inside them failed, by design). | See the next section. |

The CPU jobs ran from private work directories and the single-GPU H100 jobs ran one at a time, so the build collision did not apply to them.

## Two-node launches involving athena534

Every launch that needed Slurm to start a task on athena534 from athena533 failed with `slurm_receive_msgs: [[athena534]:6818] failed: Zero Bytes were transmitted or received`, and PMIx messages between the step daemons failed the same way. The diagnostic jobs established the following. A plain `srun --nodes=2 hostname` worked, so the nodes can run steps. `srun --mpi=pmix` failed with and without `--gpus-per-task`. `--mpi=pmi2` is rejected by this Open MPI build. Disabling UCX or limiting it to TCP did not help. `mpirun` inside the allocation failed when it tried to start its daemon on athena534 through `srun`, with the same Slurm error, and restricting its interfaces made no difference. Both nodes had an IPv4 address on `ibp103s0`, and `orted` was present on both. A two-node job on V100 nodes athena513 and athena514 worked, so this is not a limitation of two-node MPI+CUDA on Athena. It appears to be specific to athena534, which also could not create a step by itself with `--gpus-per-task`. The root cause on athena534 was not investigated further, and I have no administrator access to confirm it.

## What was not obtained

- **The timed scaling benchmark.** Job 3383610 requests 4 H100 GPUs on one node and runs the shared kernel with block 128 at 1, 2 and 4 GPUs on all three datasets with three repetitions, plus serial and single-GPU runs from the same job. It has been pending because no H100 node had four free GPUs. There are no timed scaling data, and so no multi-GPU speedup or efficiency values.
- **A two-node H100 run.** Only two-node V100 verification exists, and that checks correctness only.
- **A100 MPI results.** The A100 request never started.
- The earlier statement that "multi-GPU allocation was not obtained" would be wrong as worded. Multi-GPU allocations were obtained for verification. What was not obtained is the allocation for the timed benchmark.

## If job 3383610 runs later

It builds in the Athena checkout and writes `results/athena-mpi-h100/mpi-benchmark-*-job3383610-*`. Audit it with `GPU_NAME=H100 LAYOUTS='1x1 1x2 1x4' python3 -I tools/audit_mpi_cuda_results.py <that directory> 3383610`. The audit checks the 45-row shape, byte-identical predictions, CSV against logs, distinct GPUs per rank, full query coverage, the summary arithmetic and the Slurm state. Do not use the numbers if it fails. A new archive can be made with `python3 tools/package_athena_extra_results.py mpi <TAG>`.

## Expectation for scaling, not a result

One thread handles one query and scans every training row, and the large dataset has only 3,436 queries, so the GPU is not filled even on one device. In the 4-rank verification each rank's kernel took about 35 ms on large with 859 queries, which is the kernel time of the single-GPU shared kernel with all 3,436 queries. This suggests that adding GPUs would not shorten the kernel and that end-to-end time could grow. That is a prediction from one unrepeated run and has not been tested.

## Artifact paths

Relative to Assignment2_folder. The archive `athena-mpi-cuda-evidence-20261023.tar.gz` (with `.sha256`) holds all verification evidence, the failed-job logs, the launch diagnostics, Slurm logs, patches and source. Originals are in `HPDS_assignment2/results/athena-mpi-h100/`, `athena-mpi-v100/` and `athena-mpi/` and in `HPDS_assignment2/slurm/logs/athena-mpi-h100/` and `athena-mpi-v100/`. Submission commands are in each `submissions.txt` and in the README.
