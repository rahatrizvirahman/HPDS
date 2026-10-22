# Maple single-GPU baseline results

Completed October 22, 2026. This completes the Maple global-memory baseline milestone only. Shared memory, MPI+CUDA, Athena, CPU parallel comparisons, and the Overleaf report remain later work. No repository push or Canvas submission was made.

## Environment and allocation

Maple ran an NVIDIA GeForce GTX 1080 Ti with 11264 MiB and driver 580.178.04. The CPU is Intel Xeon E5-2690 v4 at 2.60 GHz. CUDA toolkit 12.3.107 and GCC 11.4.0 were loaded with `gcc/11 cuda/12.3`. The CUDA 13.0 field in nvidia-smi describes driver compatibility, not the compiler version used. Slurm is 25.11.1.

Partition `gpu` contains one node and one GPU. Jobs used account `egr`, QOS `normal`, `--gres=gpu:1`, one task, one CPU, and 4 GB. User aggregate limits are 2 CPUs, 1 GPU, and 16 GB. The partition maximum wall time is 30 days. No additional association job-count limit was reported. Jobs ran sequentially, with a five-minute limit for the first job and thirty-minute limits for the others.

## Build and submission

The tested base revision is `061757e079dc64b1dd6eef5e8c5917dd43561775`. The KNN sources, parser, datasets, and Makefile were unchanged from that revision. Workflow and documentation changes are captured in `HPDS_assignment2/results/final-changes.patch` and per-job patches. The archive includes the final source files.

Exact submission commands appear in the README section “Verified Maple workflow, October 22, 2026”. They explicitly set dataset selections, block sizes, k, modules, partition, account, QOS, and GPU request. The benchmark command produced job 1472. All builds executed `make -B all` inside the allocation. CPU flags were `-O2 -std=c++11 -ffp-contract=off`; CUDA flags were `-O2 -std=c++11 --fmad=false`, with `-I libarff` for both. Each Slurm log preserves expanded compilation commands and compiler versions.

## Correctness and job completion

| Job | Purpose | Final state | Exit | Elapsed |
|---|---|---|---|---|
| 1470 | Small verification and initial CUDA build | COMPLETED | 0:0 | 14 s |
| 1471 | All datasets and synthetic verification | COMPLETED | 0:0 | 22 s |
| 1472 | Full baseline benchmark | COMPLETED | 0:0 | 35 s |

All nine dataset/block combinations matched complete serial prediction files at k=3. Synthetic checks passed neighbor-order ties, class-vote ties, k=1/2/4, invalid k, and partial blocks at 1/64/128/256 threads. The serial missing-test-class case also passed. The benchmark collection tests passed in each job. No correctness check was weakened.

The benchmark contains 9 serial measurements and 27 CUDA measurements. Each of the twelve dataset/strategy/block groups has three repetitions. All 36 prediction files were independently compared byte for byte against their dataset's first serial reference after completion. Accuracies were 85.00000000%, 27.56756783%, and 99.47613525% for small, medium, and large respectively. The medium accuracy is the preserved reference result.

## Measurements

All times below are arithmetic means in milliseconds. End-to-end classification includes flattening, allocation, transfers, GPU execution, prediction retrieval, and workspace cleanup. It excludes parsing, validation, device discovery, accuracy calculation, and prediction-file output. Kernel time uses synchronized CUDA events. Each repetition launches a fresh process without explicit warm-up. Speedup is mean serial classification time divided by mean GPU end-to-end classification time.

| Dataset | Strategy | Block | Runs | Mean end-to-end ms | Mean kernel ms | Speedup |
|---|---|---|---|---|---|---|
| small | serial | 0 | 3 | 1.814725 | n/a | 1.0000 |
| small | global | 64 | 3 | 2.742088 | 2.295819 | 0.6618 |
| small | global | 128 | 3 | 2.698600 | 2.259627 | 0.6725 |
| small | global | 256 | 3 | 2.772943 | 2.323435 | 0.6544 |
| medium | serial | 0 | 3 | 124.958428 | n/a | 1.0000 |
| medium | global | 64 | 3 | 29.940981 | 27.124362 | 4.1735 |
| medium | global | 128 | 3 | 30.102791 | 27.298816 | 4.1511 |
| medium | global | 256 | 3 | 32.619582 | 29.794923 | 3.8308 |
| large | serial | 0 | 3 | 2399.946766 | n/a | 1.0000 |
| large | global | 64 | 3 | 127.003824 | 116.099424 | 18.8966 |
| large | global | 128 | 3 | 111.926179 | 100.911341 | 21.4422 |
| large | global | 256 | 3 | 115.731130 | 104.865143 | 20.7373 |

Block 128 was fastest for small and large; block 64 was fastest for medium. The small dataset was slower on the GPU. Limited query parallelism and fixed allocation/transfer costs are plausible explanations, not separately measured causes. Large/block-64 times ranged from 116.443458 to 138.463878 ms, so its variation should remain visible in later analysis. Minima and maxima for every configuration are in summary.csv. Three repetitions support a baseline comparison, not a broad statistical claim.

## Changes and limitations

No algorithm or compiler fixes were required. Verification now preserves separate dataset/block prediction files instead of deleting or overwriting them. Synthetic verification now retains per-case commands, exit status, stdout, stderr, and predictions. The job records CPU and module information, Git revision and patch, runs benchmark collector tests, and places compiler temporary files inside Assignment 2. README and personal explanations document the Maple settings.

There were no failed or incomplete Slurm jobs. The initial sandboxed cluster inspection could not create Slurm sockets. Its diagnostic output is retained; an approved inspection outside the sandbox succeeded. The missing pdftotext utility was handled by extracting the supplied PDF with installed Ghostscript. Neither issue prevented GPU work. The initial small job used the original synthetic temporary-file cleanup; full verification subsequently repeated those checks with retained artifacts. Benchmark per-process logs combine stdout and stderr, while synthetic logs store them separately.

## Artifact paths

All paths below are relative to Assignment2_folder.

- `HPDS_assignment2/results/cluster-maple.txt` preserves the sandbox inspection failure.
- `HPDS_assignment2/results/cluster-maple-unrestricted.txt` and `slurm-settings.txt` in the same directory preserve successful discovery and limits.
- `HPDS_assignment2/slurm/logs/baseline-1470.log`, `baseline-1471.log`, and `baseline-1472.log` preserve build, hardware, and execution logs.
- `HPDS_assignment2/results/jobs-accounting.txt` records final job and step states, exit codes, times, and allocated resources.
- `HPDS_assignment2/results/verify.*` contains dataset verification predictions.
- `HPDS_assignment2/results/correctness-*` contains synthetic check evidence. The generator in the archived source defines both test inputs, including the final omitted-class case.
- `HPDS_assignment2/results/baseline-20261022T195803Z-ag9hh8np` contains raw.csv, summary.csv, kernel-summary.csv, metadata.txt, and all 36 individual logs and prediction files.
- `HPDS_assignment2/results/audit.txt` records the independent measurement-count and prediction audit.
- `HPDS_assignment2/results/job-*.patch` and `final-changes.patch` preserve workflow changes relative to the tested revision.
- `maple-baseline-results-20261022.tar.gz` contains these results, logs, final source and datasets, explanations, and this summary. Its checksum is in the adjacent `.sha256` file.

No blockers remain for this baseline milestone. Maple has only one GPU, which will constrain later multi-GPU work. No measurements for later milestones are claimed here.
