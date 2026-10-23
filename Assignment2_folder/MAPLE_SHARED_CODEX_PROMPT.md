# Maple shared-memory experiment handoff

You are running in Cursor on Maple. Complete the remaining Assignment 2 shared-memory CUDA verification and paired benchmark, then preserve and package the evidence.

Read CLAUDE.md and applicable AGENTS.md first. Follow their rules. Read Assignment2_folder/NEXT_STEPS.md, ASSIGNMENT2_PLAN.md, MAPLE_BASELINE_RESULTS.md, HPDS_assignment2/README.md, and Explanations/02-shared-memory.md.

Inspect git status before updating the repository with a fast-forward pull. Preserve local changes. If the report and next-step documents are absent from the remote, tell me the local report commit needs to be pushed rather than recreating them.

The original Maple global-memory baseline is already complete and independently audited. Athena implemented and verified the shared-memory kernel. Maple still needs to verify and measure that kernel. Run global and shared together using the current source so the comparison is within one sweep. Do not rerun or overwrite the old baseline archive, modify Athena evidence, or attempt multi-GPU work on Maple.

All changes and outputs must remain within Assignment2_folder. You are authorized to inspect the queue, prepare or correct the Maple workflow, submit the required Slurm jobs, monitor them, audit the results, and create the evidence archive. Use only lecture concepts. Run compilation and computation in Slurm allocations.

First inspect Maple's queue and local results. Do not duplicate an already submitted sweep. Jobs rebuilding this checkout must run sequentially because they share executable paths.

Use the known Maple settings after confirming they still apply.

- Partition gpu, account egr, QOS normal, one GPU requested with --gres=gpu:1
- Modules gcc/11 and cuda/12.3
- One task, one CPU, 4 GB host memory
- Leave NVCCFLAGS unset to retain the Maple baseline build settings; do not inherit Athena's sm_80/sm_90 flags
- RESULTS_ROOT=results/maple-shared
- Logs in HPDS_assignment2/slurm/logs/maple-shared/
- K=3, DATASETS="small medium large", BLOCK_SIZES="64 128 256"
- STRATEGIES="global shared", TILE_ROWS=256, REPS=3

Use the exact submission commands in NEXT_STEPS.md from HPDS_assignment2. Run verification first. Require successful Slurm completion and ALL PREDICTIONS MATCH SERIAL, including synthetic partial-block, partial-tile, and tie cases. Only then submit the benchmark. Retain full predictions and logs. If anything fails, diagnose and fix within Assignment 2 without weakening checks, changing datasets, or changing the serial decision rule. Record changes and explanations.

Monitor through completion using squeue and sacct. Confirm final states and exit codes, not just successful submission. The benchmark must contain 63 measurements across 21 groups of three repetitions, comprising 9 serial and 54 CUDA runs.

Audit the actual completed directory with GPU_NAME='1080 Ti' STRATEGIES='global shared' python3 -I tools/audit_athena_results.py DIRECTORY VERIFY_JOB BENCHMARK_JOB. The tool's name is historical; its GPU and strategy parameters support this Maple sweep. Require AUDIT PASSED. Independently retain evidence of CSV/log agreement, full prediction equality, averages and speedups, GPU/compiler details, build flags, source revision/patches, and job accounting.

Write Assignment2_folder/MAPLE_SHARED_MEMORY_RESULTS.md with exact commands, environment, job IDs, correctness, three-run measurements and means, kernel times, speedups, failures or limitations, and artifact paths. Compute speedups using this sweep's serial means. Do not substitute old Maple or Athena serial times. Update the relevant personal explanations when needed.

Create a new maple-shared-memory-evidence-YYYYMMDD.tar.gz directly under Assignment2_folder, plus a .sha256 file. Include raw and summary CSVs, all run logs and prediction files, synthetic and dataset verification evidence, Slurm logs/accounting, environment, source provenance, and the summary. Include a manifest mapping archived paths to originals with SHA-256 hashes. Adapt the existing packagers into a Maple-specific tool if necessary. Verify the archive and manifest. Do not overwrite existing evidence archives.

Generated results and logs are ignored by Git. A summary-only commit will not transfer the evidence. Report the archive path and size, and identify the archive, checksum, summary, and workflow files that should be committed. Do not push until I ask.

Do not edit the Overleaf report in this run. It will be updated after these results are independently reviewed. Keep me informed of meaningful progress and finish with verification status, benchmark/audit outcomes, job IDs, artifact paths, and any remaining blockers.
