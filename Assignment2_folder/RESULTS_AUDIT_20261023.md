# Results audit and report preparation

Pulled through `7ef8264` with a fast-forward merge. Existing local documents were preserved. No cluster jobs were submitted or cancelled and no remote queue was queried during this review.

## Independent checks

All six archives pass their external SHA-256 checks and all 7,025 internal manifest entries match their recorded hashes. The report-data CSVs are byte-identical to corresponding archived CSVs.

| Evidence | Measurements | Groups of three | Result |
|---|---|---|---|
| Maple global baseline | 36 | 12 | Pass |
| Athena A100 original baseline | 36 | 12 | Pass |
| Athena A100 global/shared | 63 | 21 | Pass |
| Athena H100 global/shared | 63 | 21 | Pass |
| Athena CPU comparison | 144 | 48 | Pass |
| MPI+CUDA | No timed benchmark | Not applicable | 144 dataset verification comparisons pass |

Each benchmark row matches its individual process log, each full prediction array matches serial, and each accuracy agrees with the archived test labels within float rounding. Repetition IDs, mean/minimum/maximum times, speedups, CPU efficiencies, and kernel averages were independently checked.

For MPI+CUDA, the 36 comparisons from job 3382992, 72 from 3382997, and 36 from 3382990 match serial predictions. Their per-rank logs show distinct host/device assignments and complete ordered query coverage. Logs end with successful prediction verification. These checks establish correctness, not repeated scaling performance.

The archive for MPI+CUDA contains submissions and failed-job diagnostics but no raw current queue/accounting record for pending benchmark job 3383610. The saved prose states that it remained queued. A fresh dated scheduler snapshot is needed before finalizing that statement.

## Findings and action

- The missing Maple experiment is the shared-memory sweep. Run global and shared together with three repetitions after correctness passes. Expect 63 measurements.
- Athena's paired A100/H100 single-GPU studies and CPU comparison are complete in the supplied evidence.
- Multi-GPU resources were available for correctness checks. The unfinished work is specifically the timed 1/2/4-GPU H100 experiment.
- No multi-GPU speedup or efficiency may be claimed from the present evidence.
- No repeated multi-node timing study exists. Two-node V100 correctness evidence does exist.
- Some failed jobs were script/build or node-launch failures. They must not all be attributed to resource unavailability.
- The report presents CPU/GPU ratios from different nodes as cross-platform comparisons. Headline GPU speedups use same-sweep serial measurements.
- New measurements should retain the uncertainty shown by the original repetitions. CPU and some GPU configurations vary substantially.

## Deliverables prepared locally

- `HPDS_Assignment2_overleaf/` contains the report source, independently audited data, generated tables/figures, and reproduction tools.
- `HPDS_assignment2/Assignment_report.pdf` contains the compiled current draft.
- `HPDS_Assignment2_overleaf.zip` is the uploadable Overleaf project.
- `NEXT_STEPS.md` contains Maple run commands and Athena queue/audit commands.
- `Explanations/05-results-and-report.md` explains how evidence becomes report claims.

These are local working changes. The report remains updateable if queued results arrive before submission. The IEEE layout follows Assignment 1; no new template or page limit was supplied.
