# openmp.cpp: OpenMP

Read `00-overview-and-decomposition.md` first. This file covers only what
is specific to OpenMP: why the pragmas are structured the way they are, and
a common mistake that OpenMP makes easy to write without noticing.

## Two pragmas, not one

`#pragma omp parallel for` is a common shorthand that does two things at
once: it forks a team of threads, and it splits a loop's iterations across
that team. `openmp.cpp` deliberately does not use that shorthand. Instead
it opens the team with a plain `#pragma omp parallel` block
(`openmp.cpp:40`) and only starts splitting the loop with a separate
`#pragma omp for` (`openmp.cpp:49`) partway through that block.

The reason is that each thread needs to allocate its own private
`candidates` and `classCounts` buffers exactly once, before it starts
classifying any test instance, and reuse them for every instance it
processes (`openmp.cpp:42,44`). Fusing the two pragmas into
`parallel for` would put the loop's body, and only the loop's body, inside
the parallel construct, with nowhere to put a piece of setup code that
should run once per thread rather than once per iteration. Splitting
`parallel` from `for` opens a space between them where that per-thread
setup can go, and a matching space after the loop for per-thread teardown
(the two `free()` calls just before the closing brace).

## The mistake this design avoids

OpenMP has data-sharing clauses, `private()` and `firstprivate()`, that
sound like they should solve this directly by handing each thread its own
copy of a variable. For a pointer variable like `candidates`, they do not
do what that name suggests.

`private(candidates)` would give each thread an uninitialized `float*`,
a raw, garbage address, since `private` only reserves storage for a new
variable of the same type and never initializes it from the original.
Dereferencing that pointer crashes or corrupts memory.

`firstprivate(candidates)` would give each thread its own copy of the
*value* `candidates` held when the parallel region started, and for a
pointer, that value is an address. Every thread would end up with its own
private copy of the same address, meaning every thread would still be
reading and writing through the same shared buffer. This compiles cleanly
and often runs without crashing, which makes it a dangerous mistake: the
result is a silent data race that shows up only as accuracy drifting
between runs, exactly the same failure mode described in the overview for
sharing `candidates` between pthreads.

The rule that avoids both failures is to never put `candidates` or
`classCounts` in a data-sharing clause at all. Instead they are declared
as ordinary local variables inside the `#pragma omp parallel` block. A
variable declared inside a parallel region is private automatically,
because every thread executing that region runs the declaration itself and
gets its own independent storage for it, the same reasoning that makes
`candidates` private in the pthreads version simply by being a local
variable of a function that runs once per thread. Nothing OpenMP-specific
is needed to make it private; the ordinary scoping rules of the language
already do it.

## Why no schedule() clause is given

`#pragma omp for` (`openmp.cpp:49`) has no `schedule()` clause attached, so
it uses whatever OpenMP's `for` construct does by default. The OpenMP
standard leaves that default up to the compiler rather than fixing it, but
in practice every mainstream implementation, including the GCC runtime
used to build this project, defaults to a static split: the iteration
range is divided into contiguous chunks and one chunk is assigned to each
thread up front, with no further coordination needed while the loop runs.
Writing `schedule(dynamic)` instead would make threads request small
chunks one at a time as they finish their previous one, which helps when
different iterations take very different amounts of work.

As explained in the overview, every test instance here costs exactly the
same amount of work, `num_train * num_attributes` operations, regardless
of which instance it is. A fixed, static-style split is already balanced
without any runtime coordination, so `schedule(dynamic)` would only add
the overhead of repeatedly asking for more work with nothing to gain from
it. The default is left in place here rather than writing `schedule(static)`
explicitly, since it already does the right thing and needs no override.

## No manual capacity check needed

The pthreads version has to clamp its thread count so it never exceeds the
number of test instances, since it hands out ranges by hand before any
thread starts. `openmp.cpp` needs no equivalent check. `#pragma omp for`
distributes iterations from the loop's actual iteration count, `0` up to
`test_num_instances`, so if more threads are requested than there are test
instances, the extra threads simply receive an empty chunk and do nothing,
the same outcome the pthreads version has to guarantee by hand with its
partition formula.

## Where the barrier is

`#pragma omp for` ends with an implicit barrier by default, meaning no
thread continues past the closing brace of the loop until every thread has
finished its share of the iterations. That barrier is what makes it safe
to call `free(candidates)` and `free(classCounts)` immediately afterward
(`openmp.cpp`, just before the closing brace of the `parallel` block),
since every thread is guaranteed to be done reading and writing its own
buffers by the time any thread reaches those lines. Closing the `parallel`
block itself then joins the whole team, the same role `pthread_join` plays
explicitly in the threaded version.

`num_threads` itself never appears as a parameter to `KNN()` in this file.
`main()` calls `omp_set_num_threads(num_threads)` before parsing the
datasets, which sets the thread count for every parallel region that
follows for the rest of the program, so `KNN()` does not need to receive or
pass it along.
