# How the baseline is measured

Lecture 9, page 11, explains that kernel launches are asynchronous. A host timer stopped immediately after a launch would measure submission rather than completed GPU work. `cuda.cu` records CUDA events around the kernel and synchronizes the stop event before obtaining kernel milliseconds.

The separate host timer measures the classification operation. It includes flattened matrices, allocations, input copies, the kernel, output copies, and workspace cleanup. Parsing, input validation, device discovery, accuracy calculation, and prediction-file writes happen outside this interval. Serial timing has the corresponding classification boundary. Both programs retain their returned predictions when their timer stops.

For example, if the CPU takes 20 ms, GPU classification takes 10 ms, and the kernel takes 2 ms, the end-to-end speedup is 2. Reporting a speedup of 10 would hide GPU allocation and transfer overhead. The benchmark therefore uses CPU classification time divided by GPU classification time.

Each repetition is a fresh process with no explicit warm-up. Three repetitions produce three recorded times, and their arithmetic mean is used for the headline comparison. The minimum and maximum show some of the observed variation. The baseline runner saves raw times and predictions before producing a summary.

The job script records the host, GPU, compiler versions, and build commands. Slurm allocates the GPU before the script compiles and runs the programs. Resource options must match each cluster's configuration, following Lecture 2's resource-request workflow. The discovery script is safe to run on a login node because it does not run KNN or request GPU computation.

A correctness failure stops the sweep. Logs and prediction files remain available to explain the failure; they do not become successful timing rows. Cluster timeouts similarly leave partial evidence rather than a complete summary.

Use `k=3`, all three supplied datasets, and block sizes 64, 128, and 256 for the baseline study. The later shared-memory and multi-GPU work adds experiments rather than relabeling these baseline runs as a complete assignment.
