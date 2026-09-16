#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

void main()
{
    struct timespec start, end;
    uint64_t diff;
    
    int size = 1000;
    double *a, *b, *c;

    a = (double*)malloc(size * size * sizeof(double));
    b = (double*)malloc(size * size * sizeof(double));
    c = (double*)malloc(size * size * sizeof(double));

    //////////////////////////////////////////////////////////////////
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
        {
          double tmp = 0.0;
          
          for(int k = 0; k < size; k++)
            tmp += a[i*size + k] * b[k*size + j];
          
          c[i*size + j] = tmp;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
    printf("Sequential implementation CPU time \t\t\t\t %llu ms\n", (long long unsigned int) diff);
    
    //////////////////////////////////////////////////////////////////
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    #pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
        {
          double tmp = 0.0;
          
          for(int k = 0; k < size; k++)
            tmp += a[i*size + k] * b[k*size + j];
          
          c[i*size + j] = tmp;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
    printf("OpenMP outer loop parallelization CPU time \t\t\t %llu ms\n", (long long unsigned int) diff);
    
    //////////////////////////////////////////////////////////////////
  
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    #pragma omp parallel for
    for(int i = 0; i < size; i++)
    {        
        #pragma omp parallel for
        for(int j = 0; j < size; j++)
        {
          double tmp = 0.0;
          
          for(int k = 0; k < size; k++)
            tmp += a[i*size + k] * b[k*size + j];
          
          c[i*size + j] = tmp;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
    printf("OpenMP nested loop no collapse parallelization CPU time \t %llu ms\n", (long long unsigned int) diff);  
    
    //////////////////////////////////////////////////////////////////

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < size; i++)
    {        
        for(int j = 0; j < size; j++)
        {
          double tmp = 0.0;
          
          for(int k = 0; k < size; k++)
            tmp += a[i*size + k] * b[k*size + j];
          
          c[i*size + j] = tmp;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
    printf("OpenMP nested loop collapse parallelization CPU time \t\t %llu ms\n", (long long unsigned int) diff);  
    
    //////////////////////////////////////////////////////////////////    

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
        {
          double tmp = 0.0;
          
          #pragma omp parallel for reduction(+:tmp)
          for(int k = 0; k < size; k++)
            tmp += a[i*size + k] * b[k*size + j];
          
          c[i*size + j] = tmp;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
    printf("OpenMP inner loop parallelization CPU time \t\t\t %llu ms\n", (long long unsigned int) diff);           
    
    free(a);
    free(b);
    free(c);        
}
