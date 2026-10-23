# MPI+CUDA

`mpi_cuda.cu` runs one MPI rank per GPU. Every rank keeps the whole training set on its GPU and classifies a contiguous range of the test queries, using the same `knn_global` or `knn_shared` kernel as the single-GPU program (both now live in `knn_cuda.cuh`). A query's training scan never leaves one thread, so its neighbor order and tie rules are the serial ones and no cross-GPU merge of neighbor lists is needed. This follows the rank-and-GPU structure of the lecture's `vectorAdd_multihost_multiGPU.cu` and the partitioning in the CPU `mpi.cpp`, with one simplification. The lecture drives several GPUs from one rank, and here each rank drives exactly one.

## Partitioning and gathering

Rank `r` of `P` gets queries from `r * n / P` up to `(r + 1) * n / P`. For example, 10 queries on 4 ranks give counts 2, 3, 2, 3, and 3 queries on 4 ranks give counts 0, 1, 1, 1. An empty rank skips allocation and the kernel launch, since a zero-block launch is invalid, but it still joins every collective. Each rank copies only its slice of the test matrix to its GPU, so kernel index 0 means global query `start`. The predictions are gathered to rank 0 with `MPI_Gatherv`, as in the CPU program, because the counts differ by one when the test size does not divide evenly. They pass through host memory, so a CUDA-aware MPI build is not needed.

## Choosing each rank's GPU

The device is the node-local rank modulo the number of GPUs the task can see. That works when Slurm shows a task every GPU on the node and when `--gpus-per-task=1` shows it exactly one. A global rank must not be used as a device number, because rank 5 on the second node has no device 5. Before timing starts the ranks exchange GPU UUIDs and abort if two ranks share one, so a mistaken request for fewer GPUs than ranks fails loudly. The per-rank log line shows host, local rank, visible GPUs, PCI bus, query range and times. In the 4-rank run on athena531 the four ranks reported four different PCI buses.

## Failure handling

A rank that throws prints its message and calls `MPI_Abort`, which kills every rank. Without that, the other ranks would sit in the next collective forever. The test hook `MPI_CUDA_FAIL_RANK` makes one chosen rank fail before the first collective, and `tests/check_mpi_predictions.py` checks that the job exits nonzero within 60 seconds.

## What is timed

Each rank parses its own copy of both files before the timer starts. After a barrier the timer covers flattening, allocation, copies, the kernel, the copy back, the gather and cleanup. The reported end-to-end time is the maximum over ranks, and the kernel time is likewise the maximum of the per-rank CUDA event times. Speedup against one MPI GPU is `T(1) / T(g)` and efficiency is that divided by `g`. A worked example with made-up numbers: 40 ms on one GPU and 25 ms on two gives speedup 1.6 and efficiency 0.8.

## Pitfalls met while building it

- Two jobs that build in the same checkout at the same time overwrite each other's `./mpi_cuda`. An H100 job then failed with "no kernel image is available" while a V100 job built for `sm_70` was running, which fits the binary being replaced, and the H100 job passed when rerun alone. Jobs must run one after another.
- `${VAR:-text with {braces}}` ends at the first closing brace and leaves a stray `}` on the result. The default launch template is now set in a separate variable.
- `mpi/openmpi-4.1.8` loads gcc-toolset 13, which CUDA 12.3 does not support, so the jobs use `mpi/openmpi-4.1.6`.
- Direct launch with `srun --mpi=pmix` works only if Slurm can message every node. Launches that put a rank on athena534 failed in Slurm itself ("Zero Bytes were transmitted or received"), and that node also could not create a step with `--gpus-per-task`. `mpirun` hit the same Slurm error when it started its daemon there, so a different launcher did not avoid it. Two nodes on V100 (athena513 and athena514) worked.

## What to expect from scaling

One thread handles one query and scans every training row, so the kernel's duration is set by one thread's scan and not by how many queries share the GPU. The large dataset has only 3,436 queries, which at 128 threads per block is 27 blocks and does not fill an H100. In the 4-rank verification run each rank held 859 queries and its kernel still took about 35 ms, the same as one GPU holding all 3,436 in the single-GPU run. If the benchmark confirms this, adding GPUs will not shorten the kernel, and end-to-end time can even grow from extra process, transfer and gather costs. Whether that happens is a measurement, not a conclusion, until the scaling job has run.
