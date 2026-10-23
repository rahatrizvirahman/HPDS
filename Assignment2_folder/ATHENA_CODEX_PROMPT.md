# Athena Cursor Codex handoff

Paste the following text into the Codex chat opened on Athena.

---

We are working on HPDS Assignment 2, implementing KNN in CUDA using only concepts from the supplied lectures. You are running in Cursor on the Athena HPRC server.

I now have Athena access. For Assignment 1 I used Hickory because Athena was unavailable. Do not reuse Hickory resource settings or label Hickory results as Athena results.

You are authorized to inspect Athena's configuration, prepare Assignment 2 Slurm scripts, submit jobs, monitor them, fix problems within this scope, and generate results and logs. Run compilation and computation within appropriate Slurm allocations.

Read CLAUDE.md and any applicable AGENTS.md first, then these files.

- Assignment2_folder/ASSIGNMENT2_PLAN.md
- Assignment2_folder/HPDS_assignment2/README.md
- Assignment2_folder/HPDS_assignment2/VALIDATION.md
- Assignment2_folder/MAPLE_BASELINE_RESULTS.md
- Assignment2_folder/Explanations/
- Assignment2_folder/Assignment 2_ KNN with CUDA.pdf

Follow CLAUDE.md. Keep all new files and edits inside Assignment2_folder. Do not modify Assignment 1, lecture materials, or existing Maple evidence.

## Current status

The baseline assigns one CUDA thread per test instance and retains serial neighbor scan order and tie rules. Datasets and the ARFF parser are unchanged. Shared-memory kernels and MPI+CUDA are not implemented yet.

Maple successfully ran this baseline on a GTX 1080 Ti using CUDA 12.3 and GCC 11.4. Full prediction comparisons passed on all datasets at block sizes 64, 128, and 256. Its benchmark recorded 9 serial and 27 CUDA measurements. Do not assume Athena has the same compiler, GPU, partition, account, or QOS.

Commit d578e1d contains the Maple evidence archive. Confirm that revision or its changes are present. If the checkout is older, inspect local modifications before updating with a fast-forward pull. Never discard local work.

## Separate Athena artifacts

Use these dedicated paths, relative to Assignment2_folder.

- HPDS_assignment2/slurm/athena/ for Athena job and submission scripts
- HPDS_assignment2/slurm/logs/athena/ for Athena job logs
- HPDS_assignment2/results/athena/ for discovery, verification, benchmarks, metadata, and accounting
- ATHENA_BASELINE_RESULTS.md for the final summary
- athena-baseline-evidence-YYYYMMDD.tar.gz and its .sha256 file for the handoff archive

The current runners write to shared results locations. Before execution, add a configurable output-root option with existing defaults preserved, then configure Athena jobs to use results/athena. Apply it to verification, synthetic checks, benchmarks, patches, and temporary files. Use unique run identifiers. Record hostname, GPU model, partition, and job ID. Do not overwrite or move Maple artifacts.

## Task sequence

1. Inspect Athena with read-only commands. Use the existing cluster inspection logic and save its output under results/athena. Determine actual GPU partitions, GPU request syntax, compiler/CUDA/MPI modules, account/QOS requirements, and allocation limits. Note GPU counts and multi-node possibilities for later work, but request one GPU for this baseline. A login node need not expose a GPU.

2. Prepare an Athena-specific Slurm job or wrapper using verified settings. Inspect the existing baseline job body and preserve Maple commands. Save exact Athena commands in the README and the Athena scripts directory. Build with make -B all inside the allocation. Record complete build commands, compiler versions, hardware, source revision, and uncommitted diff. Use compatible modules and respect job quotas.

3. Start with a small-dataset build and verification job. Then verify small, medium, and large at k=3 and block sizes 64, 128, and 256. Run synthetic cases for neighbor ties, vote ties, runtime k, invalid arguments, and partial blocks. Compare every predicted class against serial. Accuracy percentages alone are insufficient. Inspect failures and fix code or scripts using lecture concepts without weakening checks or changing the supplied datasets. Preserve evidence and explain fixes.

4. After correctness passes, run K=3, DATASETS="small medium large", BLOCK_SIZES="64 128 256", and REPS=3. Record 9 serial and 27 CUDA measurements and compare all measured predictions against serial. Keep kernel time separate from end-to-end classification time. Compute speedup using Athena's own mean serial time divided by mean GPU end-to-end time. If multiple GPU models exist, select one for a comparable sweep and record it. Never pool unlike hardware into an unlabeled mean.

5. Monitor jobs using Slurm queue and accounting commands. Check final states and exit codes. Successful submission does not establish successful execution. Continue through completion when practical, preserve failure logs and partial evidence, and explain any resource blocker. Independently verify the 36 measurements, 12 groups with three repetitions, prediction files, CSV/log agreement, and summary calculations.

6. Write ATHENA_BASELINE_RESULTS.md with environment, allocations, exact commands, job IDs and final states, correctness, runtimes, speedups, changes, limitations, and artifact paths. Update personal explanations for new findings without overwriting Maple conclusions. Retain source provenance and the dataset inputs needed to interpret the evidence.

7. Create an Athena-specific evidence archive containing raw and summary CSVs, process and Slurm logs, predictions, verification cases, environment and accounting output, source provenance, and summary. Include a manifest mapping files to original paths with SHA-256 hashes, and publish an archive checksum. Adapt the Maple packager into a separate Athena tool without overwriting the Maple tool. Verify the archive and manifest.

Results and log directories are ignored by Git. Committing only the summary will not transfer evidence. Place the archive and checksum directly under Assignment2_folder so they can be explicitly included in a commit. Report their sizes and paths. Do not push or upload anything until I ask.

## Scope

Complete only the single-GPU global-memory baseline milestone on Athena. Shared-memory optimization, MPI+CUDA, parallel CPU comparisons, and the Overleaf report come later. The professor confirmed that an Excel report is unnecessary.

Keep me informed of meaningful progress. Finish with verified results, exact archive paths, and remaining blockers. Do not present this baseline as the completed assignment.
