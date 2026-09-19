# threaded.cpp: POSIX threads

Read `00-overview-and-decomposition.md` first. This file covers only what
is specific to the pthreads mechanism: how work and data reach each thread,
and why the private scratch buffers are structured the way they are.

## The problem pthread_create hands you

`pthread_create()` starts a new thread running a given function, and it can
pass that function exactly one argument: a single `void*`. Everything a
worker needs (which training and test matrices to read, which range of
test instances to own, where to write its predictions) has to be reachable
through that one pointer.

The fix used here is the `KNNArgs` struct (`threaded.cpp:22`), holding every
value a worker needs, and an array of these structs, one element per
thread. Each call to `pthread_create` is given the address of its own
element:

```cpp
pthread_create(&threads[t], NULL, KNNWorker, &args[t]);
```

so thread `t` reads `args[t]`, and only `args[t]`, for the whole time it
runs. `KNNWorker` (`threaded.cpp:34`) casts the `void*` back to `KNNArgs*`
on entry and pulls out the fields it needs.

## A bug this design avoids

A tempting shortcut is to reuse a single `KNNArgs` variable across every
`pthread_create` call in the loop, overwriting its fields each iteration.
That fails because thread creation is not instantaneous. By the time a
newly created thread actually starts running and reads its argument, the
main thread's loop may already have moved on and overwritten the struct
with the next thread's values, so two or more threads can end up reading
the same, wrong values. Giving every thread its own array element removes
the race entirely: nothing is ever overwritten while a thread might still
be reading it.

## What is shared and what is not

`args[t].train_matrix`, `args[t].test_matrix`, `args[t].num_attributes`,
`args[t].num_classes`, `args[t].train_num_instances`, and `args[t].k` are
the same for every thread and none of them is ever written after the
threads start, so every thread can read them concurrently without any
synchronization.

`args[t].predictions` is a pointer to the single shared output array, but
each thread only ever writes indices inside its own `[start, end)` range
(`threaded.cpp:116-117`), computed with the partition formula from the
overview. Since those ranges never overlap between threads, no lock is
needed around the write at `a->predictions[queryIndex] = max_class;`.

`candidates` and `classCounts` (`threaded.cpp:44,46`) are declared as local
variables inside `KNNWorker` itself, not as fields of `KNNArgs`. Because
`KNNWorker` runs once per thread, each thread executing this function body
allocates its own independent copy on its own call stack and heap
allocation. This is what makes them private: they are private not because
of any special pthreads feature, but simply because each thread is running
its own separate invocation of the function that declares them. Two
threads never see each other's `candidates` buffer, so there is no race on
the top-k list, unlike the shared-buffer bug described in the overview.

## Where the barrier is

`predictions` is not fully written until every thread has finished its
slice. The main thread enforces that by calling `pthread_join()` on every
thread it created (`threaded.cpp:125`) before returning from `KNN()`.
`pthread_join` blocks until the named thread has exited, so once the join
loop finishes, every write to `predictions` from every thread has already
happened and is visible to the main thread. This join loop is doing the
same job that OpenMP's implicit barrier at the end of a `#pragma omp for`
does automatically, but here it has to be written out explicitly.

## Reading the loop body

Once a thread has its bounds and its own private `candidates` and
`classCounts`, the loop body inside `KNNWorker` is character-for-character
the same computation as `serial.cpp`'s query loop, just reading `queryIndex`
from `a->start` up to `a->end` instead of from `0` up to
`test_num_instances`. Nothing about the distance computation, the
insertion into the sorted candidate list, or the majority vote changes.
That is deliberate: the only thing that should differ between the serial
and threaded versions is which instances each unit of execution is
responsible for, not how any single instance is classified.
