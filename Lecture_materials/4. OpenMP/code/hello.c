#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void main()
{
    //omp_set_num_threads(8);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        printf("Hello world from thread %d, total threads %d\n", tid, nthreads);
    }
}
