# Athena CPU comparison results

Completed October 23, 2026. This measures the serial, pthreads, OpenMP and MPI CPU programs on Athena with the same timing boundary as the GPU programs, so they can be compared with the GPU results. The GPU measurements were made on different nodes with different CPUs, so every cross-hardware comparison below is labeled.

## What changed in the CPU programs

`threaded.cpp`, `openmp.cpp` and `mpi.cpp` keep their KNN functions unchanged. Their `main` functions now validate input with `knn_io.h`, time with a steady clock in fractional milliseconds, count accuracy directly from the predictions (the old confusion matrix would break if a class were absent from the test set), accept an optional prediction file, and print a `RESULT` line. The MPI program reports the slowest rank and aborts all ranks if one fails. This follows what `serial.cpp` already did. The classification boundary matches the GPU programs. It starts after ARFF parsing and includes matrix flattening, the parallel work, the gather for MPI and cleanup. It excludes parsing, validation, accuracy calculation and file output.

## Environment and jobs

All jobs ran on athena201 in partition `cpu` with 16 CPUs (`--nodes=1 --ntasks=16 --cpus-per-task=1 --mem=8G --qos=normal`). The node has two AMD EPYC 7742 64-core processors (128 cores) and was not requested exclusively, so the 16 CPUs were an allocation within a larger node that other jobs may have been using. Compilers were g++ 11.5.0 and `mpicxx` from `mpi/openmpi-4.1.6`. Flags were `-O2 -std=c++11 -ffp-contract=off`, with `-pthread` for pthreads and `-fopenmp` for OpenMP. MPI ranks were launched with `srun --mpi=pmix --ntasks=N --cpus-per-task=1`, and the threaded programs were run directly in the batch step. No thread pinning was set. Each job copied the sources to a private work directory under `results/athena-cpu/` and built there, so concurrent GPU jobs could not replace its binaries.

| Job | Purpose | Final state | Exit | Elapsed |
|---|---|---|---|---|
| 3383689 | Small-dataset verification | COMPLETED | 0:0 | 1 m 52 s |
| 3383691 | All-dataset verification | COMPLETED | 0:0 | 2 m 44 s |
| 3383699 | Benchmark, 144 runs | COMPLETED | 0:0 | 4 m 10 s |

## Correctness

Synthetic cases (ties, k=1, 2 and 4, invalid k and worker counts, and more workers than test rows) passed for all three programs at 1, 2, 3, 4 and 8 workers. In job 3383691 every dataset matched the complete serial prediction file for pthreads, OpenMP and MPI at 1, 2, 4, 8 and 16 workers (k=3). All 144 benchmark prediction files matched serial.

## Benchmark

K=3, three datasets, worker counts 1, 2, 4, 8 and 16 for each parallel program, and three repetitions per configuration, each a fresh process. Times are in milliseconds. `speedup_vs_serial` divides the serial mean from this job by the configuration's mean. Efficiency is the speedup relative to the same program with one worker, divided by the worker count.

| Dataset | Program | Workers | Runs | Mean | Min | Max | Speedup vs serial | Speedup vs 1 worker | Efficiency |
|---|---|---|---|---|---|---|---|---|---|
| small | serial | 1 | 3 | 2.839216 | 2.826659 | 2.859972 | 1.0000 | n/a | n/a |
| small | pthreads | 1 | 3 | 3.460546 | 3.241679 | 3.867593 | 0.8205 | 1.0000 | 1.0000 |
| small | pthreads | 2 | 3 | 2.155963 | 1.909035 | 2.415116 | 1.3169 | 1.6051 | 0.8026 |
| small | pthreads | 4 | 3 | 1.608808 | 1.320549 | 1.803017 | 1.7648 | 2.1510 | 0.5377 |
| small | pthreads | 8 | 3 | 1.558764 | 1.245850 | 1.829677 | 1.8215 | 2.2201 | 0.2775 |
| small | pthreads | 16 | 3 | 2.642763 | 2.396971 | 2.909514 | 1.0743 | 1.3094 | 0.0818 |
| small | openmp | 1 | 3 | 2.899015 | 2.874258 | 2.947636 | 0.9794 | 1.0000 | 1.0000 |
| small | openmp | 2 | 3 | 1.840486 | 1.825459 | 1.860063 | 1.5426 | 1.5751 | 0.7876 |
| small | openmp | 4 | 3 | 1.353643 | 1.303017 | 1.431269 | 2.0975 | 2.1416 | 0.5354 |
| small | openmp | 8 | 3 | 1.366149 | 1.339947 | 1.393327 | 2.0783 | 2.1220 | 0.2653 |
| small | openmp | 16 | 3 | 64.526426 | 59.184226 | 68.509620 | 0.0440 | 0.0449 | 0.0028 |
| small | mpi | 1 | 3 | 2.921240 | 2.832440 | 3.096115 | 0.9719 | 1.0000 | 1.0000 |
| small | mpi | 2 | 3 | 4.570367 | 1.737623 | 9.980876 | 0.6212 | 0.6392 | 0.3196 |
| small | mpi | 4 | 3 | 2.785170 | 0.957478 | 6.380032 | 1.0194 | 1.0489 | 0.2622 |
| small | mpi | 8 | 3 | 5.688520 | 0.529915 | 8.284329 | 0.4991 | 0.5135 | 0.0642 |
| small | mpi | 16 | 3 | 0.475520 | 0.355878 | 0.578887 | 5.9708 | 6.1433 | 0.3840 |
| medium | serial | 1 | 3 | 198.469403 | 198.331110 | 198.653706 | 1.0000 | n/a | n/a |
| medium | pthreads | 1 | 3 | 196.542258 | 196.235535 | 196.742768 | 1.0098 | 1.0000 | 1.0000 |
| medium | pthreads | 2 | 3 | 99.981662 | 99.819614 | 100.083089 | 1.9851 | 1.9658 | 0.9829 |
| medium | pthreads | 4 | 3 | 52.153501 | 52.102826 | 52.224344 | 3.8055 | 3.7685 | 0.9421 |
| medium | pthreads | 8 | 3 | 34.325658 | 28.401502 | 45.595073 | 5.7820 | 5.7258 | 0.7157 |
| medium | pthreads | 16 | 3 | 40.279236 | 39.583573 | 40.701792 | 4.9273 | 4.8795 | 0.3050 |
| medium | openmp | 1 | 3 | 206.460092 | 205.576298 | 206.971970 | 0.9613 | 1.0000 | 1.0000 |
| medium | openmp | 2 | 3 | 104.714303 | 104.359250 | 104.985596 | 1.8953 | 1.9717 | 0.9858 |
| medium | openmp | 4 | 3 | 54.315491 | 54.107150 | 54.602841 | 3.6540 | 3.8011 | 0.9503 |
| medium | openmp | 8 | 3 | 29.402655 | 29.094282 | 29.779760 | 6.7501 | 7.0218 | 0.8777 |
| medium | openmp | 16 | 3 | 78.406000 | 63.182114 | 101.427865 | 2.5313 | 2.6332 | 0.1646 |
| medium | mpi | 1 | 3 | 218.721957 | 207.994008 | 233.173679 | 0.9074 | 1.0000 | 1.0000 |
| medium | mpi | 2 | 3 | 115.284662 | 111.810604 | 118.670929 | 1.7216 | 1.8972 | 0.9486 |
| medium | mpi | 4 | 3 | 59.820571 | 57.254501 | 63.052601 | 3.3177 | 3.6563 | 0.9141 |
| medium | mpi | 8 | 3 | 31.383569 | 30.833138 | 31.889603 | 6.3240 | 6.9693 | 0.8712 |
| medium | mpi | 16 | 3 | 18.040666 | 17.749295 | 18.543407 | 11.0012 | 12.1238 | 0.7577 |
| large | serial | 1 | 3 | 2594.672063 | 2313.990994 | 3153.928805 | 1.0000 | n/a | n/a |
| large | pthreads | 1 | 3 | 2974.284696 | 2972.781024 | 2975.783454 | 0.8724 | 1.0000 | 1.0000 |
| large | pthreads | 2 | 3 | 1854.639307 | 1852.858386 | 1856.756587 | 1.3990 | 1.6037 | 0.8018 |
| large | pthreads | 4 | 3 | 961.556199 | 960.616217 | 962.601675 | 2.6984 | 3.0932 | 0.7733 |
| large | pthreads | 8 | 3 | 507.641633 | 491.413972 | 520.078919 | 5.1112 | 5.8590 | 0.7324 |
| large | pthreads | 16 | 3 | 465.711773 | 405.053674 | 548.117230 | 5.5714 | 6.3865 | 0.3992 |
| large | openmp | 1 | 3 | 2312.296442 | 2098.301526 | 2428.484481 | 1.1221 | 1.0000 | 1.0000 |
| large | openmp | 2 | 3 | 1986.516141 | 1940.915872 | 2073.000286 | 1.3061 | 1.1640 | 0.5820 |
| large | openmp | 4 | 3 | 1009.533049 | 1002.777753 | 1022.844437 | 2.5702 | 2.2905 | 0.5726 |
| large | openmp | 8 | 3 | 539.393819 | 536.294468 | 541.353920 | 4.8103 | 4.2868 | 0.5359 |
| large | openmp | 16 | 3 | 535.540866 | 527.790318 | 541.068183 | 4.8450 | 4.3177 | 0.2699 |
| large | mpi | 1 | 3 | 2291.923906 | 2229.854104 | 2393.065070 | 1.1321 | 1.0000 | 1.0000 |
| large | mpi | 2 | 3 | 1174.592678 | 1101.960356 | 1315.009046 | 2.2090 | 1.9512 | 0.9756 |
| large | mpi | 4 | 3 | 678.410175 | 630.727732 | 772.808757 | 3.8246 | 3.3784 | 0.8446 |
| large | mpi | 8 | 3 | 420.459481 | 365.951588 | 462.711355 | 6.1710 | 5.4510 | 0.6814 |
| large | mpi | 16 | 3 | 241.171970 | 228.142303 | 262.669748 | 10.7586 | 9.5033 | 0.5940 |

Measured observations. On medium and large the parallel programs speed up clearly up to 8 workers. For example, MPI on large took 420.5 ms at 8 ranks and 241.2 ms at 16, a speedup over serial of 10.8. Pthreads and OpenMP did not improve from 8 to 16 threads on large (507.6 against 465.7 ms and 539.4 against 535.5 ms), and on medium both were slower at 16 than at 8. On small the work is so short (about 3 ms serial) that startup and variation dominate, and several results are erratic. OpenMP with 16 threads averaged 64.5 ms against 1.4 ms at 8, and MPI with 2 to 8 ranks ranged between 0.5 and 10 ms across its three repetitions. The causes were not measured. Plausible contributors are thread and process startup cost, other jobs on the same node, and the 16 CPUs not being pinned, but none of these was tested.

Run-to-run variation is large. Serial on large ranged from 2314 to 3154 ms across three repetitions, and pthreads at 8 threads on medium ranged from 28.4 to 45.6 ms. With three repetitions, speedups on the large dataset in particular carry that uncertainty, and the min and max columns should be shown next to the means.

## Comparison with the H100 GPU (different hardware)

The table divides the mean CPU time at 16 workers by the mean H100 shared-kernel time at block 128 from `ATHENA_H100_SINGLE_GPU_RESULTS.md`. The CPU runs used an AMD EPYC 7742 node and the GPU runs an H100 node with an Intel Xeon 8462Y+ host, so this compares a 16-core CPU allocation with one GPU and is not a same-machine speedup. The headline GPU speedup in the report should still use the serial time measured in the same job as the GPU run.

| Dataset | H100 shared, block 128 (ms) | pthreads, 16 threads | OpenMP, 16 threads | MPI, 16 ranks | Serial on EPYC node |
|---|---|---|---|---|---|
| small | 1.174 | 2.25 times | 54.97 times | 0.41 times | 2.42 times |
| medium | 10.261 | 3.93 times | 7.64 times | 1.76 times | 19.34 times |
| large | 42.479 | 10.96 times | 12.61 times | 5.68 times | 61.08 times |

Each entry is CPU mean time divided by GPU mean time, so values above 1 mean the GPU was faster. On small, MPI with 16 ranks (0.476 ms) beat the H100 end-to-end time (1.174 ms), the only case where a CPU program did.

## Independent audit

`python3 -I tools/audit_cpu_results.py <benchmark directory> 3383689 3383691 3383699` (output in `results/athena-cpu/audit.txt`) confirmed 144 rows in 48 groups of three repetitions, byte-identical predictions, CSV values equal to the logs, matching summary statistics, speedups and efficiencies, and all three jobs COMPLETED 0:0.

## Limitations

- One CPU node type, one allocation of 16 CPUs within a node not requested exclusively, no pinning, three repetitions, and a single node for MPI (no inter-node CPU runs).
- Variation between repetitions is large for small and for some large configurations.
- The Maple CPU comparison was not made, and these results are not comparable to the Assignment 1 timings, which used different flags and integer-millisecond timers.

## Artifact paths

Relative to Assignment2_folder. The archive `athena-cpu-evidence-20261023.tar.gz` (with `.sha256`) holds the CSVs, individual logs and predictions, verification evidence, Slurm logs, patches, audit and source. Originals are under `HPDS_assignment2/results/athena-cpu/` and `HPDS_assignment2/slurm/logs/athena-cpu/`. A copy of the CSVs is in `report_data/`.
