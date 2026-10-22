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

Record each system's successful build and verification log before collecting report measurements. Shared-memory and MPI+CUDA validation will follow their implementation in later milestones.
