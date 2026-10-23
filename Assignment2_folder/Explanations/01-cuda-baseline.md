# The first CUDA implementation

`cuda.cu` contains the host driver and `knn_global` kernel. Its indexing follows Lecture 8's vector-add example. The query index is `blockIdx.x * blockDim.x + threadIdx.x`. Each active thread handles one query and writes one output position.

With 160 test rows and 128 threads per block, two blocks launch 256 threads. Threads 0 through 159 compute predictions; the remaining threads return. This baseline uses no block barriers, so that return is safe. A later shared-memory kernel will need different handling because inactive query threads must still participate in cooperative loads and barriers.

## Host and device responsibilities

The CPU parses ARFF files and obtains flat row-major matrices. The host allocates device buffers, copies both matrices to the GPU, launches the kernel, waits for completion, and copies predictions back. This is the separate-memory workflow taught in Lecture 8. Every CUDA operation is checked, including the asynchronous kernel's completion.

The assigned Slurm task sees its allocated GPU through CUDA's visible device numbering. The baseline explicitly selects visible device 0. Later MPI work will require a separate rank-to-device mapping.

## Private candidates without a fixed k

Each query gets its own portion of the candidate-distance, candidate-label, and class-count buffers. The buffers live in global GPU memory, but their slices are logically private because only the owning query thread writes them. The workspace requires `ntest * k` distances and labels plus `ntest * classes` vote counters.

This avoids hard-coding k to three or allocating device memory inside each thread. It is a simple baseline, not a claim of optimal memory access. Threads still fetch training data repeatedly, and each query's top-k insertion is sequential. Lecture 9's memory-access discussion and Lecture 10's reuse examples motivate the next shared-memory version.

## Retaining the serial decision rule

`knn_global` accumulates features in the same order as the serial distance loop, takes the square root, and scans training instances in ascending order. It uses the same strict comparison when inserting neighbors and selecting the winning class.

GPU multiply/add fusion can change rounding. The Makefile disables contraction for the CPU distance loop and fused multiply/add for CUDA. This is a build choice to preserve the original arithmetic, not a replacement for comparison against actual serial predictions on the clusters. No fast-math optimization is enabled.

`verify_accuracy.sh` checks synthetic ties and partially filled query blocks, then compares full prediction files for small, medium, and large datasets at block sizes 64, 128, and 256. The kernel also rejects an unfilled neighbor list instead of indexing a class counter with an invalid label.

## What this version does not establish yet

This version does not use shared memory or MPI. Small datasets may provide too few queries to keep the GPU busy, and transfer/allocation costs may outweigh faster distance computation. We need measurements before deciding whether any configuration is faster than the CPU.

## Athena notes

The kernel needed no change on the A100. Block 64 had the lowest mean time on every dataset, while on Maple blocks 128 and 64 won different datasets. With three repetitions this is a baseline observation, not an explanation. The Athena build adds `-gencode arch=compute_80,code=sm_80`. Without a named architecture, nvcc 12.3 embeds code for an older target plus PTX, and the driver may compile that PTX at the first launch, which could land inside the timed kernel. The flag was added as a precaution. It was not compared against a build without it, so its effect is unmeasured.
