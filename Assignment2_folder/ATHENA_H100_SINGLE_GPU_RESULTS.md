# Athena H100 single-GPU results

Completed October 23, 2026. This repeats the global-memory and shared-memory single-GPU sweeps on an NVIDIA H100 so that the Athena MPI+CUDA work and the single-GPU results use one GPU model. The A100 results in `ATHENA_BASELINE_RESULTS.md` and `ATHENA_SHARED_MEMORY_RESULTS.md` remain valid and are kept separate. H100 and A100 numbers are never pooled.

## Environment and jobs

Both jobs ran on athena533 in partition `gpu-h100` with one NVIDIA H100 80GB HBM3 (compute capability 9.0), driver 595.91.07, `cuda/12.3` (nvcc 12.3.107) and g++ 11.4.1. The host CPU is an Intel Xeon Platinum 8462Y+. Each job used `--qos=normal --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=1 --exclude=athena534`. The build was `nvcc -I libarff -O2 -std=c++11 --fmad=false -gencode arch=compute_90,code=sm_90`, with the CPU build `g++ -O2 -std=c++11 -ffp-contract=off`. The tested sources are commit `89f20b8` plus the uncommitted changes saved in `results/athena-shared-h100/job-<ID>.patch`, which include the kernels moved into `knn_cuda.cuh`. The kernel code is the same as in the A100 shared-memory run.

| Job | Purpose | Node | Final state | Exit | Elapsed |
|---|---|---|---|---|---|
| 3382994 | Verification, both strategies, blocks 64, 128, 256 | athena533 | COMPLETED | 0:0 | 54 s |
| 3382996 | Benchmark, 9 serial and 54 CUDA runs | athena533 | COMPLETED | 0:0 | 54 s |

Submission commands are in `HPDS_assignment2/results/athena-shared-h100/submissions.txt` and in the README (`GPU_MODEL=h100 EXCLUDE=athena534 bash slurm/athena/submit_athena_shared.sh verify|benchmark`).

## Correctness

The synthetic cases (neighbor ties, class-vote ties, k=1, 2 and 4, invalid k and options, partial blocks, and 1-, 3- and 256-row tiles) passed. In job 3382994 the complete prediction files matched serial for every dataset, both strategies and blocks 64, 128 and 256 at k=3. All 63 benchmark prediction files matched serial. Accuracies are 85.00000000%, 27.56756783% and 99.47613525% for every configuration, the same as on Maple and the A100.

## Benchmark

Settings were K=3, `DATASETS='small medium large'`, `BLOCK_SIZES='64 128 256'`, `STRATEGIES='global shared'`, `TILE_ROWS=256` and three repetitions per configuration, each a fresh process with no warm-up. Timing boundaries and speedup definition are unchanged from the baseline. Times are in milliseconds. Speedup is the mean serial classification time measured in the same job divided by the mean GPU end-to-end time.

| Dataset | Strategy | Block | Runs | Mean end-to-end | Min | Max | Mean kernel | Speedup |
|---|---|---|---|---|---|---|---|---|
| small | serial | 0 | 3 | 1.043810 | 1.041589 | 1.044997 | n/a | 1.0000 |
| small | global | 64 | 3 | 1.213248 | 1.208927 | 1.216040 | 0.861792 | 0.8603 |
| small | global | 128 | 3 | 1.217383 | 1.215668 | 1.218275 | 0.867264 | 0.8574 |
| small | global | 256 | 3 | 1.224629 | 1.219881 | 1.230216 | 0.873088 | 0.8523 |
| small | shared | 64 | 3 | 1.192351 | 1.190537 | 1.195685 | 0.839435 | 0.8754 |
| small | shared | 128 | 3 | 1.173894 | 1.171738 | 1.175111 | 0.826059 | 0.8892 |
| small | shared | 256 | 3 | 1.184567 | 1.178739 | 1.192806 | 0.831275 | 0.8812 |
| medium | serial | 0 | 3 | 75.850752 | 75.710370 | 76.092924 | n/a | 1.0000 |
| medium | global | 64 | 3 | 11.754312 | 11.747264 | 11.764871 | 9.929749 | 6.4530 |
| medium | global | 128 | 3 | 11.910707 | 11.896290 | 11.918223 | 10.084288 | 6.3683 |
| medium | global | 256 | 3 | 12.256776 | 12.218370 | 12.298874 | 10.452960 | 6.1885 |
| medium | shared | 64 | 3 | 10.434058 | 10.425813 | 10.442463 | 8.604299 | 7.2695 |
| medium | shared | 128 | 3 | 10.260502 | 10.253261 | 10.265859 | 8.429472 | 7.3925 |
| medium | shared | 256 | 3 | 11.110443 | 11.088810 | 11.126684 | 9.268256 | 6.8270 |
| large | serial | 0 | 3 | 1459.903410 | 1458.180114 | 1462.095035 | n/a | 1.0000 |
| large | global | 64 | 3 | 48.321567 | 48.253130 | 48.420757 | 41.135573 | 30.2123 |
| large | global | 128 | 3 | 49.530924 | 49.437217 | 49.603275 | 42.287978 | 29.4746 |
| large | global | 256 | 3 | 51.233959 | 51.132110 | 51.322038 | 43.986187 | 28.4948 |
| large | shared | 64 | 3 | 42.779333 | 42.737266 | 42.861513 | 35.557365 | 34.1264 |
| large | shared | 128 | 3 | 42.478742 | 42.369097 | 42.688700 | 35.414090 | 34.3679 |
| large | shared | 256 | 3 | 46.863927 | 46.802877 | 46.927553 | 39.633792 | 31.1520 |

On this node the shared kernel was faster than the global kernel at the same block size in all nine dataset/block pairs, but by less than on the A100. The global kernel took 1.02 to 1.04 times as long on small, 1.10 to 1.16 times on medium and 1.09 to 1.17 times on large. Shared with block 128 had the lowest mean on all three datasets (1.174, 10.261 and 42.479 ms). On large its lead over block 64 (42.779 ms) is 0.3 ms (0.7%), and the repetition ranges do not overlap, so the lead is real but small. Kernel time is about 70% of end-to-end time on small and about 82% to 85% on medium and large for the shared kernel. Small is still slower on the GPU than serial (speedup 0.85 to 0.89), where fixed costs dominate.

The serial reference differs between platforms. The H100 job measured serial large at 1459.9 ms on a Xeon 8462Y+, while the A100 job measured 1987.5 ms on an AMD EPYC 7763. Part of any difference in speedup between the two GPUs therefore comes from the host CPU, and speedups should not be compared across the two reports. Absolute GPU times can be compared, with the labels. For example, shared block 128 on large took 42.479 ms on the H100 and 52.589 ms on the A100.

## Independent audit

`GPU_NAME=H100 STRATEGIES='global shared' python3 -I tools/audit_athena_results.py <benchmark directory> 3382994 3382996` recomputed the data from raw files (`results/athena-shared-h100/audit.txt`). It found 63 rows in 21 groups of three, byte-identical predictions, CSV values equal to the individual logs, matching summary statistics, and both jobs COMPLETED 0:0.

## Limitations

- One H100 on one node, three repetitions per configuration, and only block sizes 64, 128 and 256 with a fixed 256-row tile.
- The H100 and A100 sweeps used different host CPUs, so the speedups differ in how they were computed as well as in GPU speed.
- This sweep does not address multi-GPU behavior, which is in `ATHENA_MPI_CUDA_RESULTS.md`.

## Artifact paths

Relative to Assignment2_folder. The archive `athena-h100-single-gpu-evidence-20261023.tar.gz` (with `.sha256`) contains the CSVs, individual logs and predictions, verification evidence, Slurm logs, patches, audit and source. Original files remain in `HPDS_assignment2/results/athena-shared-h100/` and `HPDS_assignment2/slurm/logs/athena-shared-h100/`. A copy of the three CSVs is in `report_data/`.
