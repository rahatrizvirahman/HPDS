#include <stdio.h>
#include <stdatomic.h>
#include <threads.h>
 
#define NTHREADS 10
#define ITERATIONS 1e6

atomic_int atomic_counter = 0;
int regular_counter = 0;
 
int thread(void* param)
{
    for(int i = 0; i < ITERATIONS; i++) {
        atomic_counter++;
        regular_counter++;
    }
}
 
int main(void)
{
    thrd_t thr[NTHREADS];
    
    for(int i = 0; i < NTHREADS; i++)
        thrd_create(&thr[i], thread, NULL);
        
    for(int i = 0; i < NTHREADS; i++)
        thrd_join(thr[i], NULL);
 
    printf("The atomic counter is %u\n", atomic_counter);
    printf("The non-atomic counter is %u\n", regular_counter);
}
