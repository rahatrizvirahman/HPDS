#include <stdio.h>
#include <stdint.h>
#include <stdatomic.h>
#include <threads.h>

#define NTHREADS 10
#define ITERATIONS 1e6

atomic_int global_atomic_counter = 0;

void* atomic_counter(void* param)
{
    for(int i = 0; i < ITERATIONS; i++) {
        global_atomic_counter++;
    }
}

void* regular_counter(void* param)
{
    int* local_counter = (int*) param;
    
    for(int i = 0; i < ITERATIONS; i++) {
        *local_counter = *local_counter + 1;
    }
}
 
int main(void)
{
    // Structures to measure time
    struct timespec start, end;

    // Option 1. One thread using a regular counter
    int counter = 0;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    for(int j = 0; j < NTHREADS; j++) {
        for(int i = 0; i < ITERATIONS; i++) {
            counter++;
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
        
    printf("Option 1: One thread using a regular counter, result = %d\n", counter);  
    printf("Elapsed CPU time = %llu ms\n", (long long unsigned int) diff);

    
    // Option 2. Ten threads using an atomic counter
    thrd_t thr[NTHREADS];

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    for(int i = 0; i < NTHREADS; i++)
        thrd_create(&thr[i], (thrd_start_t) atomic_counter, NULL);
        
    for(int i = 0; i < NTHREADS; i++)
        thrd_join(thr[i], NULL);
        
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
        
    printf("Option 2: Ten threads using an atomic counter, result = %u\n", global_atomic_counter);  
    printf("Elapsed CPU time = %llu ms\n", (long long unsigned int) diff);
    

    // Option 3. Ten threads using respective local counters
    int counters[NTHREADS];

    for(int i = 0; i < NTHREADS; i++)
        counters[i] = 0;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    for(int i = 0; i < NTHREADS; i++)
        thrd_create(&thr[i], (thrd_start_t) regular_counter, (void *) &counters[i]);

    for(int i = 0; i < NTHREADS; i++)
        thrd_join(thr[i], NULL);

    int counter_sum = 0;
    
    for(int i = 0; i < NTHREADS; i++) {
        counter_sum += counters[i]; // Reduce results
    }
        
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
         
    printf("Option 3: Ten threads using respective local counters, result = %u\n", counter_sum);  
    printf("Elapsed CPU time = %llu ms\n", (long long unsigned int) diff);
}