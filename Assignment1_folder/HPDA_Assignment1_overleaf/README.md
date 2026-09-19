# Report (Overleaf project)

Upload this folder's contents to a new Overleaf project (or a blank
project, since `IEEEtran.cls` is included here so it compiles without
needing Overleaf's own IEEE template) and compile `main.tex` with pdfLaTeX.

Everything the report needs is in the one file, `main.tex`. It currently
compiles as-is (confirmed with `pdflatex` on maple) with placeholder
numbers, so you can see the layout before any real data is in.

Before submitting, replace:

- Your name and email in the `\author` block.
- The two dataset placeholders in the CPU/node paragraph (Section II-F) if
  your actual maple/Athena runs differ from what's written.
- Every `TODO` cell in Tables III, IV, and V with the real values from
  `results/summary.csv` (see `HPDS_assignment1/analyze_results.py`).
- The `.dat` files in `data/` with the ones `analyze_results.py` writes to
  `HPDS_assignment1/results/pgfplots/`, keeping the same filenames so the
  `\addplot` commands in `main.tex` still find them.
- The two `% TODO` paragraphs in Section IV once the real numbers show a
  specific trend to explain, and the closing sentences of Section V.

The accuracy table (Table II) and the dataset-size table (Table I) are
already filled in with real measured numbers and do not need to change
unless you rerun `verify_accuracy.sh` and get a different result.
