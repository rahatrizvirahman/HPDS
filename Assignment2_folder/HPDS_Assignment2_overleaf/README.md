# Assignment 2 Overleaf report

This is the current report draft based on completed evidence through commit `7ef8264` and the October 23, 2026 experiment snapshot. It is not a claim that all assignment experiments are complete.

## Open in Overleaf

Upload `../HPDS_Assignment2_overleaf.zip` as a new project, select `main.tex` as the main document, and compile with pdfLaTeX. The IEEE class, generated tables, plots, and data are included. Compiling the report does not require Python or the evidence archives. The local PDF was compiled with Tectonic and visually inspected; the draft has eight pages, including the individual-run appendix.

The author information and IEEE conference layout follow Assignment 1. Confirm any course-specific report length or template requirement before submission. The required Introduction, Methodology, Results, Analysis, and Conclusion sections are present. No Excel report is needed.

## Contents

- `main.tex` contains the report narrative and methodology.
- `main.pdf` is the compiled draft. A copy is in `../HPDS_assignment2/Assignment_report.pdf`.
- `data/` contains audited CSV copies, machine-readable audit records, generated tables, and figure definitions.
- `tools/audit_evidence.py` verifies the six current evidence archives from the repository without extracting or executing their contents.
- `tools/generate_tables.py` regenerates report tables and plots from `data/audited_data.json`.
- `IEEEtran.cls` is the layout class reused from Assignment 1.

To reproduce locally from this project directory within the repository, run the following.

```bash
python3 tools/audit_evidence.py
python3 tools/generate_tables.py
tectonic --keep-logs main.tex
```

The audit expects the evidence archives in the parent Assignment2_folder. It checked all 342 benchmark rows, all 7,025 manifest entries, and 144 MPI+CUDA dataset comparisons. It checks log/CSV agreement, predictions, test-label accuracy, repeated-run groups, summary arithmetic, and distinct MPI device assignments. New archive layouts or completed MPI timing data require extending the audit and table generator before adding them to this report.

## Remaining before final submission

1. Add the missing Maple paired global/shared sweep once verified.
2. Check Athena job 3383610. If it completes, audit and add the repeated MPI+CUDA scaling results and efficiency. Otherwise preserve a dated queue/accounting snapshot and update the resource-limitation statement to the actual submission cutoff.
3. Keep multi-GPU correctness distinct from multi-GPU performance. Verification already succeeded on up to four H100 GPUs and two V100 nodes.
4. Recompile and inspect the final PDF, copy it into the implementation submission folder, and refresh the Overleaf ZIP. Package the final assignment separately from the Overleaf project.

See `../NEXT_STEPS.md` for exact cluster commands. The current report does not claim an allocation remained unavailable through a deadline that has not yet passed. It also identifies the absent raw pending-job snapshot in the supplied evidence. Do not silently convert verification timing prints into repeated benchmark results.

## Build verification

The current local build has no overfull boxes or unresolved references. Two underfull bibliography lines are typesetting warnings, not missing content. Narrative pages and tables/figures were visually inspected, and extracted text was checked for unresolved reference markers. PDF content is dated to the evidence snapshot rather than a live cluster status.
