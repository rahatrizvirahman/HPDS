#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void *threadCode(void* ptr) 
{
  int tid = *(int*) ptr;

  printf("Received value in a thread: %d\n", tid);

  tid = tid = 2; // Operation by the thread
  
  pthread_exit(&tid); // incorrect way of returning a memory address (because it is local to the thread)
}
 
int main(int argc, char *argv[])
{
  int i;
  int n_threads = 4;
  pthread_t *threads;
  
  threads = (pthread_t*)malloc(n_threads * sizeof(pthread_t));

  int** results = (int**)malloc(n_threads * sizeof(int*));
  
  for(i = 0; i < n_threads; i++)
    pthread_create(&threads[i],NULL,threadCode, (void*) &i);  // incorrect way of sending respective input parameters to each thread, race condition
      
  for(i = 0; i < n_threads; i++)
    pthread_join(threads[i], (void*)&results[i]);   // waiting for threads join and collect the respective results returning from the thread

  for(i = 0; i < n_threads; i++)
    printf("Returned value by thread %d: %d\n", i, *results[i]);
  
  free(results);
  free(threads);
}
