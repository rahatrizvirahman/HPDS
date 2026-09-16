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
    
    #pragma omp parallel for
    //#pragma omp parallel for num_threads(4)
    for(int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];

        //if(omp_get_thread_num()==0) // See tid 0, 43, 44, 55
        printf("Thread %d/%d \t Index %d/%d\n", omp_get_thread_num(), omp_get_num_threads(), i, size);
    }
    
    free(a);
    free(b);
    free(c);        
}
