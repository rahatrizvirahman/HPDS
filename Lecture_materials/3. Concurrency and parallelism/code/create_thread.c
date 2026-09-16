#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void *threadCode() 
{
  // Initialize thread, load data, prepare computation
  
  // Do some cool stuff here
  
  pthread_exit(0); // terminate thread and optionally return a process-level memory adddress with results
}
 
int main(int argc, char *argv[])
{
  int i;
  int n_threads = 4;
  pthread_t *threads;
  
  threads = (pthread_t*)malloc(n_threads * sizeof(pthread_t));  // dynamic memory allocation of thread structures
  
  for(i = 0; i < n_threads; i++)
    pthread_create(&threads[i],NULL,threadCode,NULL);   // creates the threads which run immediately upon creation
      
  for(i = 0; i < n_threads; i++)
    pthread_join(threads[i],NULL);    // waits till the threads finish and optionally collect the memory addresses with results returning from the thread
  
  free(threads);
}