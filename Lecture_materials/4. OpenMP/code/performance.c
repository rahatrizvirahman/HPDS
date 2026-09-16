#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

void main()
{
    struct timespec start, end;
    uint64_t diff;

    int counter = 0;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    #pragma omp parallel num_threads(1000)
    {
        #pragma omp critical
        {
            counter++;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e3;
    printf("Critical CPU time \t %llu us\n", (long long unsigned int) diff);
    
    counter = 0;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    #pragma omp parallel num_threads(1000)
    {
        #pragma omp atomic
        counter++;
    }
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e3;
    printf("Atomic CPU time \t %llu su\n", (long long unsigned int) diff);    

    counter = 0;

    omp_lock_t lock;
	omp_init_lock(&lock);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    #pragma omp parallel num_threads(1000)
    {
        omp_set_lock(&lock);
        counter++;
        omp_unset_lock(&lock);
    }
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e3;
    printf("Locks CPU time \t\t %llu us\n", (long long unsigned int) diff);

    omp_destroy_lock(&lock);
}
