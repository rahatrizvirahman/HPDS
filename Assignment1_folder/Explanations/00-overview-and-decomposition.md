# Overview: the KNN algorithm and how the work is split

This file covers the reasoning that all three parallel versions
(`threaded.cpp`, `openmp.cpp`, `mpi.cpp`) share. Read this first; the other
three files each cover only what is specific to their own mechanism.

## What the serial algorithm does

`serial.cpp` classifies every test instance independently. For a single
test instance it:

1. Computes the Euclidean distance to every training instance
   (`distance()`, comparing all attributes except the class label).
2. Keeps a sorted list of the `k` closest training instances seen so far
   (the `candidates` array, storing distance and class for each).
3. After scanning the whole training set, counts how often each class
   appears among the `k` candidates and predicts the most frequent one.
4. Resets `candidates` and the class-count array, then moves to the next
   test instance.

So the algorithm is a doubly nested loop: for each test instance, scan every
training instance. Its cost is `O(num_test * num_train * num_attributes)`,
which is exactly what makes it worth parallelizing.

## Which loop to split

There are two independent axes to cut along, the test loop or the train
loop, and they behave very differently once split.

**Splitting the test loop** gives each worker a block of test instances and
has it scan the entire training set for each one, the same way the serial
version does. Every worker writes only its own slice of the output array
and never needs to see what another worker is doing. There is no shared
mutable state between workers at all.

**Splitting the train loop** would have every worker scan a shard of the
training set for the *same* test instance, build a partial top-k list, and
then merge the partial lists across workers to get the final top-k. This
exposes far more parallelism (the large dataset has 61,606 training
instances against only 3,436 test instances), but it changes the order in
which candidates are discovered.

That ordering matters because of how ties are broken. `serial.cpp` inserts
a new candidate only when its distance is strictly smaller than an existing
one (`if(dist < candidates[2*c])` in the original code). When two training
instances are exactly equidistant from a query, whichever one is scanned
first keeps its slot. Splitting the test loop preserves the scan order
exactly, since each worker still walks the training set from index 0 to
`n-1` for its own queries. Splitting the train loop does not, since the
final ranking now depends on the order the partial lists are merged in,
which is a different order than the original scan. An exact tie sitting at
the boundary of the top-k can then pick a different neighbor, flip a vote,
and change a prediction.

Exact ties are not a corner case here. `datasets/small-train.arff` stores
every attribute as a whole number, so squared distances are integers and
equidistant training instances are common. Since every implementation must
reproduce the serial accuracy exactly on all three datasets, splitting the
test loop is the only one of the two options that guarantees that by
construction rather than by luck. This is the reason every one of the three
parallel programs uses the same decomposition: split the test loop, keep
each worker's inner scan over the training set identical to the serial
version.

## Why the split is still well balanced

Every test instance costs exactly `num_train * num_attributes` operations
to classify, regardless of which test instance it is. That means a plain
equal-sized split of the test instances already balances the work almost
perfectly. There is no need for a dynamic work queue or a scheduler that
moves work between workers at runtime, since there is no variation in
per-instance cost for such a scheme to correct for. This is why the OpenMP
version uses `schedule(static)` rather than `schedule(dynamic)`, and why
the pthreads and MPI versions divide the test range with a fixed formula
computed once, before any work starts.

## The partition formula

All three implementations divide `n` test instances among `P` workers
(threads or MPI ranks) with the same formula. Worker `t`, numbered from 0,
owns the half-open range:

```
start = (t     * n) / P
end   = ((t+1) * n) / P
```

using integer division. This splits `[0, n)` into `P` contiguous blocks
whose sizes differ by at most one, and it absorbs whatever remainder
`n / P` leaves over automatically, so no special case is needed for the
last worker. It also degrades gracefully when `P` is larger than `n`: a
worker with `t >= n` simply gets `start == end`, an empty range, and does
no work, rather than reading past the end of the test set.

The small dataset has only 160 test instances but is run with up to 128
workers, so this last property is used in every experiment on that
dataset, not just as a defensive edge case.

## Why the output array needs no lock

Every worker writes only to the slice of `predictions[]` given by its own
`start`/`end` range, and those ranges never overlap. Two workers can run at
the same time and never touch the same array element, so no lock, atomic
operation, or critical section is needed around the write.

There is a subtler cost worth knowing about even though it does not affect
correctness. Adjacent `int` elements of `predictions[]` share the same
64-byte cache line. When one worker writes near the boundary of its block,
it can invalidate a cache line that a neighboring worker is also touching,
a pattern called false sharing. A contiguous block split like the one used
here has exactly one such boundary per pair of adjacent workers, which is
a negligible cost next to the actual distance computations. An interleaved
split, where worker 0 took test instances 0, 8, 16, and so on, would put a
shared cache line between every single pair of writes and would be
measurably slower for no benefit to load balance. This is the second
reason (alongside tie-break determinism) that the partition uses
contiguous blocks rather than any interleaved pattern.

## One shared trap: state that looks read-only but is not

`candidates` and the class-count array are reused across test instances in
the serial code: allocated once, filled in, read, and then reset in place
for the next query. That reuse is safe in serial code because there is only
ever one query being processed at a time. As soon as more than one worker
is active, two workers sharing the same `candidates` buffer would
overwrite each other's in-progress top-k list, and the resulting accuracy
would drift and change from run to run depending on how the operating
system happened to schedule the threads.

Each of the three parallel versions solves this the same way: every worker
allocates its own private copy of `candidates` and the class-count array,
once, before it starts processing its slice of test instances, and reuses
that private copy across all of the queries it owns. The three files
implement "private" differently (a struct field per pthread, a variable
declared inside an `omp parallel` block, or a plain local variable in an
MPI process that already has its own address space), and each file's own
explanation covers the mechanism in detail.

## Measuring the parallel versions

Two definitions of speedup are used in the report, since they answer
different questions.

**Absolute speedup**, `S(p) = T_serial / T_parallel(p)`, compares a
parallel run against the unmodified serial program. This is the number
that reflects a user's actual experience: it charges the parallel version
for every overhead it introduces, including thread or process creation and
any imbalance from the partition.

**Relative speedup**, `S(p) = T_parallel(1) / T_parallel(p)`, compares a
parallel version against its own single-worker run instead of against
`serial.cpp`. This isolates how well a given implementation scales as
workers are added, but it hides any fixed overhead that the parallel
version pays even at one worker, since that overhead appears in both the
numerator and the denominator.

Both are reported for every configuration. The report treats absolute
speedup as the headline number, since it is the one that answers whether
parallelizing was worth doing at all, and uses relative speedup to discuss
how cleanly each version scales once its fixed costs are set aside.

The estimated parallelizable fraction of the code is computed with the
Karp-Flatt metric, from the absolute speedup `S` at `p` workers:

```
e = (1/S - 1/p) / (1 - 1/p)
parallel fraction = 1 - e
```

where `e` is the estimated serial fraction. Unlike Amdahl's law solved in
reverse, Karp-Flatt is meant to be evaluated at more than one value of `p`
and compared: if the estimated serial fraction grows as `p` increases, that
growth is itself evidence of parallel overhead (synchronization, load
imbalance, or communication) that a fixed-fraction model like Amdahl's law
cannot see, since Amdahl's law assumes the serial fraction is a constant
property of the code rather than something that can grow with the amount
of overhead added by using more workers.
