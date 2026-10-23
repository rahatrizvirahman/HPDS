# Independent review of Maple baseline evidence

Reviewed the evidence pulled in commit `d578e1d`. The missing-artifact issue is resolved by `maple-baseline-evidence-20261022.tar.gz` and its checksum file.

## Checks completed

- The archive SHA-256 matches the published checksum, and all 292 manifest entries match their individual hashes.
- The raw CSV contains 36 measurements across 12 groups, each with repetitions 1, 2, and 3. These comprise 9 serial and 27 CUDA runs.
- All 36 measured prediction arrays match their dataset's serial reference byte for byte.
- Recomputed accuracies using the archived test labels match the recorded accuracies within float rounding.
- Each measurement's strategy, runtime, kernel runtime where applicable, and accuracy agree with its individual process log.
- All summary means, minima, maxima, speedups, and kernel means recompute correctly from raw measurements.
- Retained dataset verification outputs match serial. Retained synthetic cases have the expected predictions and exit status, including invalid k values.
- Slurm accounting records jobs 1470, 1471, and 1472 and their listed steps as completed with exit code 0:0. Verification and benchmark logs record successful completion.
- Archived implementation files match the checkout. The README differs because the checkout includes later packaging instructions.

## Verified findings

| Dataset | Best observed block size | Mean GPU classification ms | Speedup over serial |
|---|---|---|---|
| Small | 128 | 2.698600 | 0.6725 |
| Medium | 64 | 29.940981 | 4.1735 |
| Large | 128 | 111.926179 | 21.4422 |

These are the best observed means across three repetitions, not proof of a universally optimal block size. Large/block-64 times vary from 116.443458 to 138.463878 ms. Preserve that variation in reporting.

This review checks archived evidence and recomputes its results. It does not rerun GPU computation. The verified scope is the Maple single-GPU global-memory baseline. Athena, shared memory, MPI+CUDA, parallel CPU comparisons, and the final report remain subsequent work.
