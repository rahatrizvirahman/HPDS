# Assignment 2 implementation and report plan

## Scope and current status

This document records the overall plan. The first CUDA baseline milestone is now implemented in `HPDS_assignment2/`, with serial prediction checks, a configurable Slurm job, baseline measurements, and personal explanations. As of October 23, 2026 the baseline and shared-memory kernels, the MPI+CUDA program and the CPU comparison are implemented. Athena has A100 and H100 single-GPU results, CPU comparison results and MPI+CUDA verification, but no timed multi-GPU scaling run. Maple has baseline results only. The Overleaf report remains to be written. See `REPORT_HANDOFF.md`. Existing starting materials were committed first in `2e49982`.

All new work belongs under `Assignment2_folder/`. Treat `Assignment1_folder/` and `Lecture_materials/` as reference material. Follow the root `CLAUDE.md` rules in code comments, explanations, and report prose.

The assignment explicitly permits extending the previous assignment's code. Reuse the same KNN classification problem, ARFF parser, and small, medium, and large train/test pairs. The four copied CPU implementations and all six datasets were checked and are identical to Assignment 1.

| Dataset | Training instances | Test instances | Features excluding class |
|---|---|---|---|
| Small | 1,184 | 160 | 7 |
| Medium | 14,708 | 740 | 11 |
| Large | 61,606 | 3,436 | 11 |

Keep `k=3` as the experiment default inherited from Assignment 1. The Assignment 2 PDF does not specify a new value. Preserve support for a valid runtime `k`, or explicitly document any supported limits rather than silently hard-coding three neighbors.

## Confirmed workflow

The professor confirmed that an Excel report is not required, as relayed on October 22, 2026. Prepare the Overleaf report and final PDF. Keep CSV files as experiment records and inputs for analysis and figures.

The user will clone the repository on Maple and Athena and execute the code there using Slurm commands. Prepare reproducible build instructions, platform-specific Slurm scripts, and commands for submission, monitoring, and result collection. Run commands from the cloned `Assignment2_folder/HPDS_assignment2/` directory and keep generated logs and results within Assignment 2. The user will return the logs and CSV files for analysis and report preparation.

Before finalizing job scripts, use the cluster configuration reported by the user to confirm partitions, GPU requests, modules, account/QOS requirements, and available GPU counts. Do not assume the two systems share the same Slurm configuration.

## Requirements taken from the assignment

The primary source is [Assignment 2 PDF](<Assignment 2_ KNN with CUDA.pdf>). Page 1 describes the work; pages 2–4 give the rubric. The saved PDF shows October 25, 2026 as the due date, without a submission time.

| Requirement | Planned evidence |
|---|---|
| Build single-GPU CUDA and MPI+CUDA programs | Makefile targets and successful builds and runs on all three datasets |
| Match serial accuracy | Checks for both GPU program families on all datasets, plus checks for each delivered strategy and configuration |
| At least two distinct kernel strategies | Global-memory baseline and shared-memory training-tile version, with explanations of their differences |
| Maple results on all three datasets | Accuracy and runtime logs for each submitted strategy |
| Athena results | Correct Slurm scripts, three individual runs and their averages for every submitted CUDA version on all datasets |
| Kernel configuration experiments | Both strategies at multiple block sizes, three repetitions per dataset/configuration |
| MPI+CUDA execution | Each rank uses its assigned GPU, processes its own work, and contributes to correctly ordered predictions |
| Multi-GPU scaling | Three GPU counts including one GPU and two distinct larger counts, three repetitions on all datasets |
| Multi-node evidence | Multi-node runs or a documented resource limitation |
| CPU comparisons and speedup | Serial, pthreads, OpenMP, and MPI CPU comparisons using explicitly defined timing boundaries |
| Multi-GPU efficiency | Speedup relative to one GPU and efficiency relative to GPU count |
| Report | Introduction, Methodology, Results, Analysis, Conclusion, with labeled tables and figures |
| Submission archive | Source, scripts, datasets, Makefile, and report |

Use the rubric's explicit requirement to run all three datasets on both named systems. Do not interpret the ambiguous word “respectively” on page 1 as assigning different datasets to different machines.

## Lecture concepts allowed in the implementation

Use the supplied lectures and their code examples as the technical basis. The proposed KNN kernels adapt those concepts to the existing algorithm rather than introducing a new nearest-neighbor algorithm.

| Source under `Lecture_materials/` | Concepts used here |
|---|---|
| Lecture 8, `8. Introduction to CUDA-1/8. Introduction to CUDA.pdf`, pp. 9–16; `code/vectorAdd.cu` and `code/Makefile` | Grid/block/thread indexing, bounds checks, host/device buffers, memory copies, kernel launches, `nvcc` |
| Lecture 9, `9. CUDA memory access pattern-1/9. CUDA memory access pattern.pdf`, pp. 2–11; `code/matrixAdd.cu` | Memory access patterns, coalescing, stride, indexing, synchronized CUDA event timing |
| Lecture 10, `10. CUDA shared memory-1/10. CUDA shared memory.pdf`, pp. 2–10; `code/matrix_multiplication_9_complete_tiles.cu` | Shared-memory allocation, cooperative tile loading, reuse, block synchronization |
| Lecture 11, `11. CUDA reduction and sorting/11. CUDA reduction and sorting.pdf`, pp. 2–16 | Background for reduction and sorting tradeoffs; not required for the first two kernels |
| Lecture 13, `13. CUDA asynchronous streams and multi-GPU-1/13. CUDA asynchronous streams and multi-GPU.pdf`, pp. 2–4 and 17–25 | Occupancy/resource tradeoffs, device selection, multi-GPU and MPI organization |
| `vectorAdd_multihost_multiGPU.cu`, Lecture 6 MPI material, and copied `mpi.cpp` | Rank partitioning, communication, collecting results, coordinating GPU work |
| `2. Slurm for HPC clusters-1.pdf` | Resource requests, batch jobs, and job logs |

The lecture MPI+CUDA example uses a rank per host controlling multiple GPUs. Adapt its device-selection and partitioning ideas to the rubric's simpler one-rank-per-GPU design. Gather predictions through host memory using the MPI pattern already present in Assignment 1.

Streams are taught in Lecture 13 but are optional follow-up work only if measurements justify them. The initial scope needs neither an external KNN library nor a new sorting implementation. Do not add techniques beyond the supplied lectures. Apply bounds checks and synchronization carefully rather than copying assumptions from fixed-size lecture examples.

## Proposed folder structure

The layout below describes the target structure. Baseline, shared-memory, MPI+CUDA and CPU files now exist; report files remain planned. The current cluster workflow uses a shared configurable job body, with platform-specific resource options supplied to `sbatch`. Historical helpers are archived; the new baseline runner is `benchmark_baseline.py`.

```text
Assignment2_folder/
  ASSIGNMENT2_PLAN.md
  Assignment 2_ KNN with CUDA.pdf
  Explanations/                       personal learning notes, excluded from submission
  Reference_assignment1/              copied historical results and report, if archived
  HPDS_assignment2/
    serial.cpp                       preserved CPU reference
    threaded.cpp, openmp.cpp, mpi.cpp CPU comparison implementations
    cuda.cu                          two selectable single-GPU strategies
    mpi_cuda.cu                      MPI driver using the same GPU strategies
    knn_cuda.cuh                      shared GPU declarations/helpers if needed
    libarff/, datasets/               reused inputs and parser
    Makefile, README.md
    verify_accuracy.sh
    bench_common.sh, run_experiments.sh, analyze_results.py
    slurm/                           Assignment 2 GPU jobs, collection, logs
    results/                         fresh Assignment 2 measurements and plots
    Assignment_report.pdf            final Assignment 2 report, generated later
  HPDS_Assignment2_overleaf/          report sources and figure data, created later
```

## Ordered work plan

### 1. Prepare the copied folder before changing the algorithm

- Inventory the copied `README.md`, report PDF, results, plots, and Slurm logs. They still describe Assignment 1 and must not be presented as Assignment 2 evidence.
- Archive those historical artifacts under `Assignment2_folder/Reference_assignment1/` before preparing fresh result directories. Keep useful CPU code and benchmark helpers in the implementation folder.
- Rewrite the Assignment 2 README with the eventual program interfaces, build instructions, correctness checks, and execution workflow.
- Preserve the parser and datasets. Keep the existing serial algorithm as the correctness reference.
- Check the actual Maple and Athena CUDA compiler, MPI installation, GPU models and memory, GPU availability, and Slurm allocation limits before choosing build architecture flags or resource requests. Record these observations for the report. Old Hickory settings are historical information, not verified Athena settings.

Completion evidence is a clearly identified baseline, clean destinations for new measurements, and a confirmed GPU build/run environment.

### 2. Establish correctness and timing rules

Retain Euclidean distance over features only, the existing float arithmetic and feature accumulation order, ascending training-instance scan order, strict insertion comparisons for equal-distance neighbors, and the smallest class index when class votes tie. Do not assume removing the square root or rearranging arithmetic preserves all predictions.

Create an optional prediction-output/checking path in the Assignment 2 copies so each GPU prediction can be compared with the serial prediction. Compare full predictions as well as reported accuracy. Equal accuracy percentages can conceal different classifications.

Check all datasets, partial blocks, partial training tiles, valid `k` bounds, and uneven MPI partitions. GPU compiler arithmetic can differ from CPU arithmetic even with the same source order. Investigate any mismatch at the distance and neighbor-selection level before accepting timings, and document any arithmetic/build adjustment needed to retain the reference behavior.

Define timing before collecting data. Report synchronized kernel time separately from end-to-end classification time. The primary classification timer starts after ARFF parsing and includes matrix preparation, allocations, host-to-device transfers, all GPU work, device-to-host predictions, and required cleanup. Keep equivalent boundaries for CPU comparison runs. State separately that file parsing and accuracy calculation are excluded, matching the previous classification-focused measurement. If whole-process elapsed time is also reported, label it separately.

Use a host wall-clock timer for end-to-end time and lecture CUDA events for kernel time. Preserve fractional milliseconds, since the copied helpers currently parse integers. For MPI+CUDA, include data distribution and result collection, synchronize the measured region, and report the maximum rank elapsed time. Never divide CPU end-to-end time by GPU kernel-only time as the headline speedup.

### 3. Implement the first CUDA strategy

Assign one CUDA thread to one test instance. Each active thread scans all training instances in their original order, computes distances, maintains its own top-k candidates and class counts, and writes one prediction.

Reuse the flattened matrices from the ARFF parser and explicitly copy inputs and outputs between host and device. Give each query independent workspace, sized for the supported `k` and class count. Check allocation sizes, CUDA calls, kernel launches, and completion errors.

This strategy uses Lecture 8 indexing and memory transfers and provides a straightforward correctness baseline. It may expose too little parallel work on the small dataset. Record that limitation rather than assuming it will be fast.

Add `nvcc` build targets alongside the CPU targets. Establish successful prediction checks on all datasets before moving on.

### 4. Implement the shared-memory strategy

Keep one query per thread and the same neighbor-selection logic. A block cooperatively loads a bounded tile of training instances into shared memory using adjacent loads. After synchronization, its active query threads reuse that tile while processing training instances in ascending order. Synchronize again before replacing the tile.

Keep inactive threads in a partial query block participating in all required block barriers. Guard tile loads and process only valid rows in the final tile. Do not return early before a barrier used by the rest of the block.

Choose tile sizes that fit the device's shared-memory limits for both feature counts. Initially hold tile size fixed while varying block size so the comparison is interpretable. Account for private candidate storage and shared-memory consumption when discussing occupancy.

Compare this strategy with the baseline using identical datasets, `k`, timing boundaries, and correctness checks. Shared-memory reuse adds synchronization overhead, so decide which version is best from measurements.

### 5. Add MPI+CUDA

- Use one rank per allocated GPU and explicitly select that rank's device. Respect the devices visible inside each Slurm task. Do not use a global MPI rank as an unchecked local CUDA device index.
- Log rank, hostname, and device assignment so that distinct GPU use can be demonstrated.
- Retain the full training set on every participating GPU. Divide test queries into contiguous ranges using `start = rank * n / ranks` and `end = (rank + 1) * n / ranks`.
- Reuse the single-GPU kernel implementation for each rank's local query range. This avoids a distributed top-k merge and preserves each query's training scan order.
- Return local predictions to host memory and gather them in their original test order using the existing MPI variable-count gathering pattern. Handle empty ranges and remainder queries correctly.
- Validate the one-GPU MPI path first, then multiple GPUs and multiple nodes. Ensure a failed rank cannot leave the others indefinitely waiting in a collective.

### 6. Run the required experiments

Use three independent measured repetitions for every required configuration. If warm-up is used, label it and apply the same documented policy throughout. Save individual observations before computing averages.

| Experiment | Planned sweep |
|---|---|
| Single-GPU kernel comparison | 2 strategies × block sizes 64, 128, 256 × 3 datasets × 3 repetitions on Maple and Athena where supported |
| MPI+CUDA scaling | GPU counts 1, 2, 4 as the initial target × 3 datasets × 3 repetitions, using a justified kernel configuration |
| Multi-node comparison | At least one feasible configuration spanning nodes, with GPU count and placement recorded |
| CPU comparisons | Serial, pthreads, OpenMP, and MPI on the comparison platforms with recorded worker counts and consistent timing, three repetitions |

The proposed single-GPU sweep is 54 measured runs per platform. A single selected MPI+CUDA strategy at three GPU counts adds 27 runs per platform where that sweep is available. If more MPI+CUDA strategies are delivered, include their required runs too.

Confirm GPU counts before scheduling. If four GPUs are unavailable but three are allowed, use 1, 2, 3. Three distinct GPU counts remain a rubric requirement. Documenting a multi-node limitation does not automatically waive the separate GPU-count requirement.

Extend the copied scripts to record platform, hostname, GPU model, strategy, dataset, `k`, block size, tile size, GPU count, rank count, node count, repetition, end-to-end time, kernel time, accuracy, and correctness status. Record compiler versions and flags with the experiment metadata. Keep unsuccessful runs visible and exclude them from successful-run averages.

Prepare separate Maple and Athena Slurm scripts using each system's verified GPU allocations and modules. Provide `sbatch` submission commands, `squeue` monitoring commands, and instructions for collecting logs and CSV files. The user will run these commands on the clusters. Collect raw logs and all three repetitions into fresh CSV files. Rerun CPU baselines for fair comparisons where possible. If old Assignment 1 results are included for context, identify their hardware, flags, dates, and timing boundaries and label cross-machine comparisons explicitly.

### 7. Analyze results

Compute the arithmetic mean over the three successful repetitions and retain the individual values. Show variation, such as the minimum and maximum, alongside the average where useful.

Use `S_serial = T_serial / T_GPU_end_to_end`. Also compare `T_pthreads / T_GPU_end_to_end`, `T_OpenMP / T_GPU_end_to_end`, and `T_MPI_CPU / T_GPU_end_to_end` at stated CPU worker counts.

For GPU scaling, use the same MPI+CUDA implementation and settings for `S_GPU(g) = T_MPI_CUDA(1) / T_MPI_CUDA(g)` and `E_GPU(g) = S_GPU(g) / g`. Do not substitute CUDA thread count for GPU count in this efficiency calculation.

Identify the best strategy and block size for each dataset and platform. Explain observed trends using lecture concepts such as memory traffic, reuse, synchronization, occupancy, available query parallelism, transfers, and MPI overhead. Separate measured evidence from explanations that are plausible but unmeasured. Do not copy the old CPU worker-count analysis or Karp-Flatt tables into GPU results without a specific justified need.

## Personal explanations folder

Maintain `Assignment2_folder/Explanations/` alongside implementation work, as in Assignment 1. These notes are for personal understanding and stay outside the submission folder and ZIP.

| Planned note | What it should explain |
|---|---|
| `00-overview-and-reuse.md` | What carries over, the KNN steps, tie rules, and why queries can be processed independently |
| `01-cuda-baseline.md` | Mapping queries to threads, indexing, host/device memory, private candidates, and prediction checks |
| `02-shared-memory.md` | Why tiling may help, cooperative loading, barriers, partial tiles, and resource costs |
| `03-mpi-cuda.md` | Rank/GPU assignment, query partitioning, replicated training data, and gathered predictions |
| `04-timing-and-experiments.md` | Timing boundaries, synchronization, repetitions, Slurm requests, and raw-result fields |
| `05-results-and-report.md` | How each plot is produced, speedup calculations, observed limitations, and report claims |

For every note, explain both why the design was chosen and how the implementation works. Include a small worked example, references to the relevant lecture pages and code functions, and pitfalls actually encountered. Update notes when a design changes so they describe the delivered implementation.

## Later Overleaf report and submission

Create `Assignment2_folder/HPDS_Assignment2_overleaf/` after the methodology and experiment layout are stable. Reuse the previous report's LaTeX layout only as a starting point, subject to confirmation of the required report format. Replace all Assignment 1 narrative, figures, and measurements.

Write the required sections in order. Introduction describes the KNN task and GPU motivation. Methodology explains both kernels, MPI partitioning, correctness, timing, and hardware/software. Results presents individual Athena runtimes, averages, accuracy, kernel configurations, CPU comparisons, and GPU scaling. Analysis explains the measured trends. Conclusion states the supported findings and limitations.

Generate labeled tables and plot data from the recorded CSVs and keep units, strategies, hardware, and configuration labels explicit. Include multi-GPU efficiency and multi-node results or evidence of the limitation. Compile the report and inspect it before copying the final PDF into `HPDS_assignment2/`.

Package source, required headers/parser files, datasets, Makefile, scripts, and report into a fresh Assignment 2 ZIP. Verify a clean extracted copy builds and that report figures trace back to raw measurements. Exclude personal explanations, archived Assignment 1 artifacts, and unrelated binaries.

## Information needed before later execution or submission

Nothing else is required to create this plan. Resolve these items before the corresponding later phase.

1. Obtain the Maple and Athena configuration outputs from the user before finalizing Slurm scripts and scheduling experiments. Confirm usable GPU counts, partitions, modules, and GPU job limits. The user will clone the repository and execute the jobs on both systems.
2. Confirm any required report template or page limit. The visible PDF requires a format but does not identify it. The Assignment 1 IEEE-style Overleaf project is a provisional layout, not proof of the Assignment 2 requirement.
3. Share any separate starter archive or instructor clarification if one exists. None is needed to begin from this copy because the assignment expressly permits reuse.

The first implementation milestone is preparation of the copied folder followed by a correct single-GPU baseline. Shared memory, MPI+CUDA, the full measurements, and the Overleaf report follow in that order.
