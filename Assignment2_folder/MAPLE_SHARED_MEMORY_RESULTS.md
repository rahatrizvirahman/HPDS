# Maple shared-memory CUDA results

Completed October 23, 2026. The paired global/shared experiment passed verification and completed all 63 requested measurements. This fills the Maple shared-memory experiment gap. It does not add multi-GPU measurements or update the Overleaf report.

## Checkout and preserved evidence

The requested `git pull --ff-only` stopped because local cleanup commit `82b9ec3` diverged from `origin/main`. After the user authorized fixing the divergence, merge commit `2481ddac6694e842b819e8cde41db498eceada60` preserved both branches and incorporated `a750972`. Two documentation conflicts were resolved by retaining both Maple cleanup and Athena sections. No algorithm conflict or source correction was needed. CUDA sources, serial code, datasets, parser and Makefile match `a750972`.

Both jobs tested the merge revision above. Per-job status and patches, source hashes, and changes relative to the handoff are retained. The new audit and packaging tools and final documentation were added during this run; they do not change the measured executables. Existing Maple and Athena archives passed all seven available checksum checks and were not overwritten. The Overleaf files match the handoff revision without edits. Nothing was pushed.

## Environment and settings

Maple uses a GeForce GTX 1080 Ti with 11264 MiB, driver 580.178.04, and an Intel Xeon E5-2690 v4 at 2.60 GHz. Modules `gcc/11` and `cuda/12.3` provided GCC 11.4.0 and CUDA compiler 12.3.107. The CUDA 13.0 label in nvidia-smi describes driver compatibility, not the compiler used.

Jobs requested partition `gpu`, account `egr`, QOS `normal`, one GPU with `--gres=gpu:1`, one node, one task, one CPU, and 4 GB. The user association allows aggregate resources of two CPUs, one GPU, and 16 GB. The partition maximum wall time is 30 days. The queue was empty before submission; accounting showed only the original baseline jobs and no previous shared-memory sweep. Jobs ran sequentially because they rebuild the same executable paths.

Both jobs used k=3, all three datasets, block sizes 64/128/256, both global and shared strategies, and shared tile size 256 rows. The benchmark used three repetitions. `NVCCFLAGS` was explicitly unset. The Makefile supplied `-O2 -std=c++11 --fmad=false` for CUDA and `-O2 -std=c++11 -ffp-contract=off` for serial, with `-I libarff` for both. No Athena architecture flags were inherited.

## Exact submission and audit commands

Run from `Assignment2_folder/HPDS_assignment2`. The first job was confirmed COMPLETED with exit 0:0 and ALL PREDICTIONS MATCH SERIAL before the second submission.

```bash
mkdir -p slurm/logs/maple-shared results/maple-shared
unset NVCCFLAGS
MODULES='gcc/11 cuda/12.3' RESULTS_ROOT=results/maple-shared K=3 DATASETS='small medium large' BLOCK_SIZES='64 128 256' STRATEGIES='global shared' TILE_ROWS=256 sbatch --parsable --partition=gpu --account=egr --qos=normal --gres=gpu:1 --time=00:20:00 --output=slurm/logs/maple-shared/verify-%j.log slurm/baseline_job.sh verify
# Returned job 1560
unset NVCCFLAGS
MODULES='gcc/11 cuda/12.3' RESULTS_ROOT=results/maple-shared K=3 DATASETS='small medium large' BLOCK_SIZES='64 128 256' STRATEGIES='global shared' TILE_ROWS=256 REPS=3 sbatch --parsable --partition=gpu --account=egr --qos=normal --gres=gpu:1 --time=00:30:00 --output=slurm/logs/maple-shared/benchmark-%j.log slurm/baseline_job.sh benchmark
# Returned job 1561 after 1560 was COMPLETED 0:0 with ALL PREDICTIONS MATCH SERIAL

sacct -X -j 1560,1561 --format=JobID,State,ExitCode,Elapsed,AllocTRES%80 > results/maple-shared/jobs-accounting.txt
GPU_NAME='1080 Ti' STRATEGIES='global shared' python3 -I tools/audit_athena_results.py results/maple-shared/baseline-20261023T204556Z-job1561-ezlt9y_4 1560 1561
python3 -I tools/audit_maple_shared_evidence.py
```

Each job ran `make -B all` inside its Slurm allocation. The expanded build commands were identical in both logs.

```bash
g++ -I libarff -O2 -std=c++11 -ffp-contract=off serial.cpp libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp -o serial
nvcc -I libarff -O2 -std=c++11 --fmad=false cuda.cu libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp -o cuda
```

## Correctness, accounting, and audit

| Job | Purpose | State | Exit | Elapsed | Requested limit |
|---|---|---|---|---|---|
| 1560 | Global/shared verification | COMPLETED | 0:0 | 36 s | 20 min |
| 1561 | Paired benchmark | COMPLETED | 0:0 | 49 s | 30 min |

All 18 dataset/strategy/block verification prediction files matched serial. Synthetic verification covered neighbor-order ties, class-vote ties, k=1/2/4, invalid k, partial query blocks, shared tiles of 1/3/256 rows, and invalid strategy/tile options. It preserved 165 command/exit records, per-case stdout/stderr, and 61 prediction files. The omitted-test-class check is serial only. Both batch jobs passed the three benchmark-collector tests.

The required audit printed AUDIT PASSED. There are exactly 63 measurements, comprising 9 serial, 27 global, and 27 shared runs. All 21 groups contain repetitions 1, 2, and 3. Every full prediction file is byte-identical to its dataset's serial reference. All CSV values agree with the RESULT records in individual logs. Means, minima, maxima, repetition counts, and speedups agree with independent recomputation. Kernel times were checked separately. The supplementary audit verified expected prediction counts, block sizes, tile settings, GPU/compiler details, build flags, unchanged tested source hashes, and the presence of both successful accounting records. It printed ADDITIONAL AUDIT PASSED.

Accuracies for both kernels and serial were 85.00000000% on small, 27.56756783% on medium, and 99.47613525% on large. No correctness checks or datasets were changed.

## Measurements

All times below are milliseconds. The three end-to-end columns are the individual fresh-process runs. Kernel triples show the corresponding individual CUDA event times. Each mean is the arithmetic mean of the three runs. There was no explicit warm-up. End-to-end classification includes flattening, allocation, transfers, completed GPU work, output retrieval, and workspace cleanup. Parsing, validation, device discovery, accuracy calculation, and file output are outside the timer. Headline speedup divides this sweep's serial mean by the GPU end-to-end mean; no old Maple or Athena serial timing is used.

| Dataset | Strategy | Block | E2E r1 | E2E r2 | E2E r3 | Mean E2E | Kernel r1 / r2 / r3 | Mean kernel | Serial speedup |
|---|---|---|---|---|---|---|---|---|---|
| small | serial | 0 | 1.810196 | 1.799203 | 1.849955 | 1.819785 | n/a | n/a | 1.0000 |
| small | global | 64 | 2.953018 | 2.951615 | 2.941801 | 2.948811 | 2.503680 / 2.509824 / 2.508800 | 2.507435 | 0.6171 |
| small | global | 128 | 2.936543 | 2.946776 | 2.952162 | 2.945160 | 2.503680 / 2.508800 / 2.518848 | 2.510443 | 0.6179 |
| small | global | 256 | 3.049438 | 3.074964 | 3.114362 | 3.079588 | 2.604032 / 2.627584 / 2.665472 | 2.632363 | 0.5909 |
| small | shared | 64 | 2.816179 | 2.835117 | 2.812057 | 2.821118 | 2.376704 / 2.386944 / 2.373440 | 2.379029 | 0.6451 |
| small | shared | 128 | 2.813317 | 2.812066 | 2.817069 | 2.814151 | 2.367488 / 2.382848 / 2.379776 | 2.376704 | 0.6467 |
| small | shared | 256 | 2.869840 | 2.864844 | 2.854994 | 2.863226 | 2.433024 / 2.413376 / 2.410496 | 2.418965 | 0.6356 |
| medium | serial | 0 | 124.861282 | 124.673034 | 124.650843 | 124.728386 | n/a | n/a | 1.0000 |
| medium | global | 64 | 32.918688 | 32.825487 | 32.983722 | 32.909299 | 30.138176 / 30.054399 / 30.171167 | 30.121247 | 3.7901 |
| medium | global | 128 | 32.869919 | 32.787582 | 32.720493 | 32.792665 | 30.061567 / 29.984800 / 29.915136 | 29.987168 | 3.8035 |
| medium | global | 256 | 35.780483 | 35.458132 | 35.457788 | 35.565468 | 32.967617 / 32.671745 / 32.613377 | 32.750913 | 3.5070 |
| medium | shared | 64 | 30.556625 | 30.555010 | 30.522612 | 30.544749 | 27.732832 / 27.752447 / 27.702272 | 27.729184 | 4.0835 |
| medium | shared | 128 | 30.535491 | 30.683432 | 30.551919 | 30.590281 | 27.751137 / 27.754496 / 27.749376 | 27.751670 | 4.0774 |
| medium | shared | 256 | 32.765973 | 32.880761 | 32.822410 | 32.823048 | 29.965183 / 30.030848 / 30.001152 | 29.999061 | 3.8000 |
| large | serial | 0 | 2399.768300 | 2399.187480 | 2400.185141 | 2399.713640 | n/a | n/a | 1.0000 |
| large | global | 64 | 148.984103 | 134.283317 | 123.186379 | 135.484600 | 135.776230 / 123.414207 / 112.423393 | 123.871277 | 17.7121 |
| large | global | 128 | 117.640269 | 117.559853 | 117.602292 | 117.600805 | 106.891968 / 106.815422 / 106.846977 | 106.851456 | 20.4056 |
| large | global | 256 | 123.026065 | 123.100335 | 123.057500 | 123.061300 | 112.287682 / 112.308029 / 112.286430 | 112.294047 | 19.5001 |
| large | shared | 64 | 111.447585 | 111.382681 | 111.236372 | 111.355546 | 100.653091 / 100.615837 / 100.475777 | 100.581568 | 21.5500 |
| large | shared | 128 | 111.486410 | 111.521092 | 111.642823 | 111.550108 | 100.741501 / 100.714371 / 100.839394 | 100.765089 | 21.5124 |
| large | shared | 256 | 116.071636 | 116.138361 | 116.194560 | 116.134852 | 105.311615 / 105.388481 / 105.430878 | 105.376991 | 20.6632 |

Shared memory had a lower mean end-to-end time than global memory in all nine matched dataset/block comparisons. The best shared configurations were block 128 on small (2.814151 ms, 0.6467× serial), block 64 on medium (30.544749 ms, 4.0835×), and block 64 on large (111.355546 ms, 21.5500×). Small remains slower on the GPU than on serial CPU.

Large/global/block-64 varied from 123.186379 to 148.984103 ms, so its mean comparison should be interpreted with that variation visible. Large/shared blocks 64 and 128 differ by only about 0.195 ms in their three-run means. These measurements identify the fastest tested means, not a statistically established universal optimum. Training-tile reuse is a plausible explanation for the shared kernel's improvement, but no profiler measured the cause. Tile size was held at 256 and was not optimized.

## Failures and limitations

No Slurm job, compilation, correctness check, or measurement failed. The initial fast-forward pull was blocked by branch divergence and was resolved through the authorized merge. No implementation fix or benchmark workflow change was needed. The supplementary audit and Maple-specific packager add evidence checks and transfer support only. No multi-GPU work was attempted on Maple. There are no remaining blockers for this experiment.

## Evidence and transfer

All new working evidence is under `HPDS_assignment2/results/maple-shared/`. The original Maple baseline remains under `results/maple-baseline/`; Athena evidence is preserved separately.

- `results/maple-shared/baseline-20261023T204556Z-job1561-ezlt9y_4/` contains raw.csv, summary.csv, kernel-summary.csv, metadata.txt, 63 individual logs, and 63 prediction files.
- `results/maple-shared/verify.job1560.*` and `correctness-job1560.*` contain full dataset and synthetic verification evidence.
- `slurm/logs/maple-shared/verify-1560.log` and `benchmark-1561.log` contain compiler commands, hardware information, and all job output.
- `results/maple-shared/jobs-accounting.txt`, `job-steps-accounting.txt`, and `accounting-machine.txt` preserve final job states and exit codes.
- `results/maple-shared/audit.txt` and `additional-audit.txt` preserve both successful audits.
- `results/maple-shared/preflight.txt`, `submissions.txt`, `tested-revision.txt`, `tested-source.sha256`, per-job patches/status, and `final-changes.patch` preserve environment and source provenance.
- `results/maple-shared/prior-archive-checksums.txt` records successful checks of existing evidence archives.

The new `maple-shared-memory-evidence-20261023.tar.gz` is directly under `Assignment2_folder`, alongside its `.sha256` file. Its internal manifest maps every copied archived path to its original path and SHA-256 digest. The archive groups summaries, raw measurements, predictions, logs, verification, environment, provenance, and source. It includes `knn_cuda.cuh` and the datasets. It omits empty temporary directories, binaries, and Python caches. The packager verifies all copied bytes and manifest contents after reopening the archive. No duplicate extracted bundle is left in the results folder.

Create and check the archive with these commands from `HPDS_assignment2`.

```bash
python3 tools/package_maple_shared_results.py
cd ..
sha256sum -c maple-shared-memory-evidence-20261023.tar.gz.sha256
```

A complete evidence commit must include the archive, checksum, this summary, `HPDS_assignment2/tools/audit_maple_shared_evidence.py`, `HPDS_assignment2/tools/package_maple_shared_results.py`, and the updated README, VALIDATION, NEXT_STEPS, and shared-memory explanation. Ignored result directories are not transferred by a normal source commit. The merge commit must also be retained. No new experiment commit or push was made in this run; only the merge was committed to repair the checkout.
