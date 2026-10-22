# How the baseline is measured

Lecture 9, page 11, explains that kernel launches are asynchronous. A host timer stopped immediately after a launch would measure submission rather than completed GPU work. `cuda.cu` records CUDA events around the kernel and synchronizes the stop event before obtaining kernel milliseconds.

The separate host timer measures the classification operation. It includes flattened matrices, allocations, input copies, the kernel, output copies, and workspace cleanup. Parsing, input validation, device discovery, accuracy calculation, and prediction-file writes happen outside this interval. Serial timing has the corresponding classification boundary. Both programs retain their returned predictions when their timer stops.

For example, if the CPU takes 20 ms, GPU classification takes 10 ms, and the kernel takes 2 ms, the end-to-end speedup is 2. Reporting a speedup of 10 would hide GPU allocation and transfer overhead. The benchmark therefore uses CPU classification time divided by GPU classification time.

Each repetition is a fresh process with no explicit warm-up. Three repetitions produce three recorded times, and their arithmetic mean is used for the headline comparison. The minimum and maximum show some of the observed variation. The baseline runner saves raw times and predictions before producing a summary.

The job script records the host, GPU, compiler versions, and build commands. Slurm allocates the GPU before the script compiles and runs the programs. Resource options must match each cluster's configuration, following Lecture 2's resource-request workflow. The discovery script is safe to run on a login node because it does not run KNN or request GPU computation.

A correctness failure stops the sweep. Logs and prediction files remain available to explain the failure; they do not become successful timing rows. Cluster timeouts similarly leave partial evidence rather than a complete summary.

Use `k=3`, all three supplied datasets, and block sizes 64, 128, and 256 for the baseline study. The later shared-memory and multi-GPU work adds experiments rather than relabeling these baseline runs as a complete assignment.

## Maple validation workflow

The confirmed Maple request uses one GPU in `gpu`, account `egr`, QOS `normal`, and modules `gcc/11 cuda/12.3`. Small verification precedes full verification, which precedes the measured sweep. All compilation and KNN execution happen inside Slurm allocations.

Verification originally deleted its temporary predictions. It now retains them with dataset and block identifiers so a mismatch can be inspected after the job ends. Synthetic checks likewise retain each command, exit status, stdout, stderr, and prediction file. For example, the small dataset at block size 64 has its own file rather than being overwritten by block size 128. The job records the Git revision and a patch because uncommitted workflow changes are not identified by the revision alone. Temporary compiler files stay under Assignment 2 through `TMPDIR`.

## Athena validation workflow

Athena has different settings from Maple, so none were copied. The login node has no CUDA, which meant the toolkit and GPU models had to be found from inside short one-GPU jobs. Those jobs showed `cuda/12.3`, `12.9` and `13.2` modules, A100, H100 and V100 partitions, and no account requirement. The sweep used one A100 and the `cuda/12.3` module (the same toolkit version as Maple). Lecture 2's resource-request workflow applies here, with the partition and `--gres=gpu:1` taken from the cluster rather than from memory.

Evidence from the two clusters must not share files. Each script now reads `RESULTS_ROOT`, so Athena writes to `results/athena` while the default stays `results`. A `RUN_ID` such as `job3382505` is added to directory names, which keeps two jobs from reusing a name. For example, `verify.job3382504.XXXXXX` and `verify.job3382502.XXXXXX` can sit side by side.

The audit tool recomputes every number from the raw logs and prediction files, so it does not simply trust the collector that produced the CSV. It found 36 rows, 12 groups of three, byte-identical predictions and matching summaries. A check written by a different program is better evidence than the program checking itself.

Athena speedups divide Athena serial time by Athena GPU end-to-end time. The A100 hosts a different CPU from Maple, so a serial time from Maple would mix hardware. On large, the A100 end-to-end time was 62.4 ms at block 64 against 1987.6 ms serial, a speedup of about 31.8. On small the GPU was slower (0.87), consistent with the Maple finding that 160 queries leave the GPU mostly idle and fixed costs dominate.
