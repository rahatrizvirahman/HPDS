# The shared-memory kernel

`knn_shared` in `cuda.cu` keeps the baseline's decomposition of one query per thread. What changes is where a thread reads training rows from. In the global kernel every thread fetches every training row from global memory. In the shared kernel the threads of a block load a tile of training rows into shared memory once and then all scan it. This follows Lecture 10, pages 2 to 10, and the tiled matrix multiplication example, where a block cooperatively loads a tile, synchronizes, computes from the tile, synchronizes again and moves on.

## How a block works

Take a block of 4 query threads, 6 training rows and a tile of 4 rows. The block makes two passes. In the first, 4 rows are loaded. Thread 0 copies the first float of the tile, thread 1 the second, and so on, which keeps adjacent threads on adjacent addresses as Lecture 9 recommends. A barrier follows so no thread reads a half-loaded tile. Each active thread then scans the 4 rows in order and updates its own candidates. A second barrier stops any thread from racing ahead while others still read the tile. In the second pass only 2 rows remain, so the last tile is partial. The loader copies `rows * attributes` floats, and the scan loop runs `rows` times, so nothing outside the real rows is read.

Because each thread still sees training rows in ascending order and uses the same `row_distance`, `insert_candidate` and `choose_class` helpers as the global kernel, the tie rules and predictions are unchanged. Splitting the scan into tiles does not reorder it.

## Barriers and partial query blocks

With 160 queries and 128 threads, the second block has only 32 active threads. If inactive threads returned early they would never reach the barriers and the block would hang or read stale tiles. The kernel therefore sets an `active` flag and lets every thread run the load and barrier steps. Only the distance scan and the final write are guarded by `active`. This is the main difference from the baseline, where returning early was safe because it had no barriers.

## Choices and limits

The tile is 256 rows by default (`--tile`). On the 12-column datasets that is 12288 bytes, which fits comfortably in one block's shared memory. The host checks `tile_rows * attributes * 4` against the device limit and rejects a tile that does not fit. Tile size was held fixed while block size varied, so the measurements say nothing about other tile sizes.

Each thread's candidate distances, labels and vote counts remain in global memory. Only the training rows moved. A larger gain might need those moved too, but that is not needed for this milestone.

## What the measurements showed

On the Athena A100 the shared kernel was faster than the global kernel for every dataset and block size, with the global kernel taking 1.10 to 1.28 times as long end-to-end. On large at block 64 that is 52.6 ms against 63.9 ms. The small dataset still did not beat serial (0.93). The likely reason is reuse of training rows across the block, but no profiler was used, so this is an explanation and not a measured cause.

## Pitfall found while testing

A tile of 1 or 3 rows against 4 training rows is the cheapest way to hit partial tiles and many tile passes. The synthetic test in `tests/check_predictions.py` uses these, with block sizes of 1 and 2 to leave the last query block partial, since the real datasets rarely hit those edge cases.

## Maple measurements

Maple verified both strategies with the current source in job 1560, then measured them together in job 1561. The paired sweep has 9 serial and 54 CUDA runs, with all predictions matching serial. Shared memory had lower mean end-to-end time at every matching block size. On large at block 128, global averaged 117.600805 ms and shared averaged 111.550108 ms. This is a modest improvement despite reusing the training tile, because candidate buffers still use global memory and each query still scans all training rows in order. That explanation follows the memory-access design; no profiler was used to establish the cost breakdown.

The best shared mean was 111.355546 ms on large at block 64, giving 21.5500 times speedup against this sweep's 2399.713640 ms serial mean. Small still took longer on the GPU. Tile size was held at 256 rows, so this experiment does not identify an optimal tile size. All three repetitions, kernel times, and timing variation are retained in MAPLE_SHARED_MEMORY_RESULTS.md and the evidence archive. Original Maple baseline times were not substituted into the new comparisons.
