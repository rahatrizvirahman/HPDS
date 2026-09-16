#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void main()
{
    int size = 100;
    int *a, *b, *c;

    a = (int*)malloc(size * sizeof(int));
    b = (int*)malloc(size * sizeof(int));
    c = (int*)malloc(size * sizeof(int));
    
    #pragma omp parallel
    {
        int id, nthrds, elements_per_thread, start, end;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();
        elements_per_thread = (size + nthrds - 1) / nthrds; // Warning: integer division used correctly

        start = id * elements_per_thread;       
        end = (id+1) * elements_per_thread; 
        if (end > size) end = size; // Now multiple threads may have no workload

        //if(omp_get_thread_num()==0) // See tid 0, 49, 50, 55
        printf("Thread %d/%d \t Range %d-%d\n", omp_get_thread_num(), omp_get_num_threads(), start, end);

        for(int i = start; i < end; i++) {
            c[i] = a[i] + b[i];            
            //printf("Thread %d/%d \t Index %d/%d\n", omp_get_thread_num(), omp_get_num_threads(), i, end);
        }
    }
    
    free(a);
    free(b);
    free(c);        
}
