# How the current report uses the evidence

The report is in `HPDS_Assignment2_overleaf/`. Its offline audit reads the six evidence archives without extracting or executing their contents. It verifies external checksums, internal manifest hashes, full prediction equality, accuracy against test labels, process logs, and CSV summary arithmetic. The 342 measured runs are kept separate by hardware and sweep.

Tables and plots are generated from the audited data rather than copied by hand from prose summaries. Every individual classification time is included in the appendix. This makes it possible to trace a displayed mean to all three original measurements. The original A100 global baseline remains separate from the later paired global/shared sweep because their source versions and measured runtimes differ.

The headline GPU speedup uses the serial time measured in the same sweep. The separate EPYC CPU comparison can still be compared with the H100, but its ratio is labeled as a cross-platform comparison because the CPU host and allocation differ. CPU efficiency uses the same CPU implementation with one worker as its baseline.

Multi-GPU correctness and multi-GPU performance answer different questions. The saved MPI predictions establish correct classification on up to four GPUs and two nodes. One verification run per configuration cannot provide the required three-run scaling averages. The report therefore describes those checks without turning their timing prints into benchmark results.

The outstanding allocation is for the timed H100 scaling experiment, not all multi-GPU work. The saved handoff reports job 3383610 pending, but the archive lacks its current raw queue/accounting snapshot. The report dates that statement to the evidence cutoff and the next-step document asks for the fresh snapshot. We cannot yet say resources were unavailable before the deadline because that deadline is still in the future.

The missing Maple experiment is the shared-memory sweep. Running both global and shared kernels together will provide a comparison using the same source and allocation while preserving the original global-only evidence. Once either missing sweep completes, audit it before regenerating the report.
