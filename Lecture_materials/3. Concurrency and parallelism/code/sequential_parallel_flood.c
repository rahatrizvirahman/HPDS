#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>

void *threadCode() 
{
  // Initialize thread, load data, prepare computation
  
  // Do some cool stuff here
  
  pthread_exit(0);
}
 
int main(int argc, char *argv[])
{
  int i, j;
  int n_threads = 4 * 1000; 
  pthread_t *threads;
  struct timespec start, end;
  
  threads = (pthread_t*)malloc(n_threads * sizeof(pthread_t));
  
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  // Create ALL threads and run them at the same time (measure impact of creating an excessive number of threads)
  for(i = 0; i < n_threads; i++)
    pthread_create(&threads[i],NULL,threadCode,NULL);
    
  for(i = 0; i < n_threads; i++)
    pthread_join(threads[i],NULL);

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  
  uint64_t diff = 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  
  printf("Elapsed process CPU time = %llu miliseconds\n", (long long unsigned int) (diff / 1e6));
  
  free(threads);
}