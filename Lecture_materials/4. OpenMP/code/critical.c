#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

void main()
{
    struct timespec start, end;
    uint64_t diff;

    int critical_counter = 0;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    #pragma omp parallel num_threads(1000)
    {
        #pragma omp critical
        {
            critical_counter++;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e3;
    printf("Critical CPU time \t %llu us\n", (long long unsigned int) diff);
        
    int atomic_counter = 0;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    #pragma omp parallel num_threads(1000)
    {
        #pragma omp atomic
        atomic_counter++;
    }
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e3;
    printf("Atomic CPU time \t %llu us\n", (long long unsigned int) diff);
    
    printf("Counter critical %d\n", critical_counter);
    printf("Counter atomic %d\n", atomic_counter);
}
