#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>

sem_t sem_main, sem_threads;
unsigned char killthreads = 0;

void *threadCode() 
{
  // Initialize thread, load data, prepare computation
  
  while(1)
  {
      // Wait for synchronization
      sem_wait(&sem_threads);
      
      if(killthreads)  pthread_exit(0);
      
      // Do some cool stuff here
          
      // Tell the main thread you're done (for now)
      sem_post(&sem_main);
  }
}
 
int main(int argc, char *argv[])
{
  int i, j, n_threads = 4;
  pthread_t *threads;
  struct timespec start, end;
  
  threads = (pthread_t*)malloc(n_threads * sizeof(pthread_t));
  
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  sem_init(&sem_main, 0, 0);
  sem_init(&sem_threads, 0, 0);

  for(i = 0; i < n_threads; i++)
    pthread_create(&threads[i],NULL,threadCode,NULL);
    
  // Create threads one and keep them alive waiting for workload
  for(j = 0; j < 1000; j++)
  {
    for(i = 0; i < n_threads; i++)
      sem_post(&sem_threads); // signal threads to run
      
    for(i = 0; i < n_threads; i++)
      sem_wait(&sem_main);    // wait for threads
      
    // Do some sequential stuff  
  }
  
  killthreads = 1;
  
  for(i = 0; i < n_threads; i++)
      sem_post(&sem_threads); // signal threads to run  

  for(i = 0; i < n_threads; i++)
    pthread_join(threads[i],NULL);
    
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  
  uint64_t diff = 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  
  printf("Elapsed process CPU time = %llu miliseconds\n", (long long unsigned int) (diff / 1e6));
  
  free(threads);
}