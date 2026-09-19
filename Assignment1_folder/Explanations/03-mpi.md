# mpi.cpp: MPI

Read `00-overview-and-decomposition.md` first. This file covers what is
specific to MPI: why processes need a different data strategy than threads,
and the one genuinely new step this version has that the other two do not,
collecting results back into a single array at the end.

## Processes, not threads

The pthreads and OpenMP versions run inside one process. Every worker
shares the same address space, so a pointer computed on one thread means
the same thing on every other thread, and `train_matrix` or `predictions`
can simply be handed to every worker by reference.

MPI ranks are separate operating system processes, typically one per CPU
core, each with its own private address space. A pointer in rank 0's
memory is meaningless in rank 3's memory. Anything one rank needs that
another rank produced has to be explicitly sent over, through an MPI call,
rather than just read through a shared pointer. This one fact is the
reason the MPI version needs a data-distribution strategy and a collection
step that the threaded versions do not.

## Getting the dataset to every rank without sending it

`main()` (near the top of the file) has every rank independently call
`ArffParser::parse()` on both the training and test files, before
`MPI_Init` even returns control past the setup calls. Since every rank runs
this same code, every rank ends up with its own complete, independently
parsed copy of both datasets in its own memory, without any rank ever
transmitting a byte of the dataset to another rank.

This trades memory for communication. Every rank holds the full training
and test matrices, so if a dataset were too large to fit in one node's
memory, this approach would not scale. For the datasets in this
assignment, the training and test matrices are small enough that
redundant per-rank memory use is not a concern, and the benefit is that
the only communication anywhere in the program becomes the one step that
collects the final predictions back together at the end, discussed below.
Parsing also happens before the timer starts (`clock_gettime` is called
after `MPI_Barrier`, later in `main()`), so this redundant work is not part
of the measured runtime at all.

## Every rank computes the whole partition table

Inside `KNN()`, every rank builds two arrays of length
`mpi_num_processes`, `counts` and `displs`, using the same partition
formula from the overview (`mpi.cpp:47-50`):

```cpp
displs[r] = r_start;             // where rank r's block starts, globally
counts[r] = r_end - r_start;     // how many test instances rank r owns
```

Every rank runs this same loop over every rank index `r`, not just its own,
so every rank ends up knowing the full table, not only the slice it owns
itself. That is what `my_start = displs[mpi_rank]` and
`my_count = counts[mpi_rank]` then pull out for this rank's own use
(`mpi.cpp:53-54`).

This looks redundant, since only rank 0 will actually need the full table
(as explained below), but computing it independently on every rank is
cheap: it is a loop over at most 128 numbers doing simple arithmetic. The
alternative, computing it once and sending it to every other rank, would
require its own communication step to save an amount of computation that
is not worth saving.

If `mpi_num_processes` is larger than the number of test instances, some
ranks get `counts[r] == 0`, the same graceful degradation the partition
formula gives the pthreads version. A rank with zero test instances simply
skips its own classification loop (`my_count` is `0`, so the `for` loop
never executes) and still participates correctly in the collection step
described next, contributing zero elements.

## Classifying, locally

Each rank allocates its own private `candidates` and `classCounts`
(`mpi.cpp:60-62`), the same private scratch buffers described in the
overview, and its own `local_predictions` array sized to exactly its own
`my_count` (`mpi.cpp:59`). It then runs the same query loop as the serial
version, but writing results into `local_predictions[i]` at a local index
`i` running from `0` to `my_count`, rather than into a global
`predictions[queryIndex]` (`mpi.cpp:70-104`). There is no data race to
avoid here at all: unlike threads, no other rank can read or write this
rank's `local_predictions`, `candidates`, or `classCounts`, since they
exist in this rank's own address space and nowhere else.

## Collecting the results: the step unique to MPI

At this point every rank holds only its own slice of the answer, in its
own private memory. Rank 0 is the one that needs the complete
`predictions` array, since only rank 0 computes the confusion matrix and
prints the result (`main()`, guarded by `if (mpi_rank == 0)`). Getting
every other rank's slice into rank 0's array is what `MPI_Gatherv` does:

```cpp
MPI_Gatherv(local_predictions, my_count, MPI_INT,
            predictions, counts, displs, MPI_INT,
            0, MPI_COMM_WORLD);
```

Every rank calls this same line. Each rank sends its own `local_predictions`
buffer, `my_count` elements from it. The root argument, `0`, tells every
rank where to send its data. On rank 0 only, the `predictions`, `counts`,
and `displs` arguments are used as the receive side: rank 0 places rank
`r`'s incoming `counts[r]` elements starting at offset `displs[r]` in
`predictions`. Since `displs[r]` was defined to be exactly the global test
index where rank `r`'s block starts, every rank's contribution lands
exactly where it belongs, and `predictions` comes out in the same order it
would have if one process had computed the whole thing serially.

`MPI_Gatherv`, the varying-count form, is used instead of the plain
`MPI_Gather` because `MPI_Gather` requires every rank to send the same
fixed number of elements. The partition formula gives blocks that differ
by at most one element whenever the test set does not divide evenly by the
number of ranks (3,436 test instances over, say, 8 ranks is 429 or 430
each), so the count genuinely does vary rank to rank, and `MPI_Gatherv`'s
per-rank `counts` argument is what makes that possible.

`MPI_Gatherv` is also a synchronization point: no rank returns from that
call until every rank has arrived at it and finished sending its data.
That means the timing measurement on rank 0 (`clock_gettime` called right
after `KNN()` returns) reflects however long the slowest rank took to
finish its own classification loop, not just rank 0's own share of the
work. With the partition formula giving every rank almost exactly equal
work, no single rank should be a consistent straggler, but this is why an
`MPI_Barrier` is placed before the timer starts in `main()` as well
(`mpi.cpp:188`): without it, a rank that happened to finish parsing its
input files earlier than the others would start its own clock earlier and
then sit idle inside `MPI_Gatherv` waiting for the others, and that idle
waiting time would be counted as part of its own measured runtime.

## What changes at higher rank counts, and why the report distinguishes it

Threads communicate through shared memory automatically. MPI ranks
communicate only through explicit calls like `MPI_Gatherv`, and how
expensive that call is depends on where the ranks physically are. Ranks on
the same machine exchange data through memory, which is fast. Ranks on
different machines exchange data over the network, which is slower. This
is the reason the assignment (and `slurm/knn_job.sh`, see its own README
section) distinguishes MPI runs forced onto a single Athena node from runs
allowed to spread across multiple nodes: the algorithm and the code are
identical in both cases, but the physical cost of the one `MPI_Gatherv`
call at the end can differ once ranks are talking over the network instead
of within one machine's memory.
