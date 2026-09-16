#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>

void *threadCode(void* arg) 
{
  int * x = (int*) arg;

  *x += 1;
  
  pthread_exit(0);
}
 
int main(int argc, char *argv[])
{
  int * x = malloc (sizeof(int));
  int i, n_threads = 10000;
  pthread_t *threads;
  
  threads = (pthread_t*)malloc(n_threads * sizeof(pthread_t));

  *x = 0;

  for(i = 0; i < n_threads; i++)
    pthread_create(&threads[i],NULL,threadCode,x);
      
  for(i = 0; i < n_threads; i++)
    pthread_join(threads[i],NULL);

  printf ("%d\n", *x);

  free(x);
  free(threads);
}
