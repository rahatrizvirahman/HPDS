#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>

#define BUFFER_SIZE 10

typedef struct 
{
  int data[BUFFER_SIZE];
  int in;
  int out;
} buffer;

buffer my_buffer;
sem_t full, empty, mutex;

void *producer(void *param) 
{
  int i, err, myid;

  myid = atoi((char *)param);
  i = 0;
  
  while(1)
  {
      /* Sleep for random time */ 
      sleep(random() % 5);		

      sem_wait(&empty);
      sem_wait(&mutex); 

      /* Add item to buffer */
      my_buffer.data[my_buffer.in] = (int) random() % 100;  
      
      fprintf(stdout,"Producer %d, Item %d produced: %d\n", myid, i, my_buffer.data[my_buffer.in]); 
      i++;

      /* Update in pointer */
      my_buffer.in = (my_buffer.in + 1) % BUFFER_SIZE;
      
      sem_post(&mutex);
      sem_post(&full);
  }

  pthread_exit(0);
}

void *consumer(void *param) 
{
  int err, myid;

  myid = atoi((char *)param);

  while(1)
  {
      sem_wait(&full);
      sem_wait(&mutex); 

      fprintf(stdout," \t\t\t\t Cons: %d consumed: %d\n", myid, my_buffer.data[my_buffer.out]);
	  
      /* Update out pointer */
      my_buffer.out = (my_buffer.out + 1) % BUFFER_SIZE;

      /* Sleep for random time */ 
      sleep(random() % 5);
      
      sem_post(&mutex);
      sem_post(&empty);
  }
  
  pthread_exit(0);
}

 
int main(int argc, char *argv[])
{
  pthread_t tid1, tid2, tid3, tid4;

  sem_init(&full, 0, 0);
  sem_init(&empty, 0, 1);
  sem_init(&mutex, 0, 1);

  /* Initialize random seed */
  srand(time(NULL));

  /* create the threads */
  pthread_create(&tid1,NULL,consumer, "1");
  pthread_create(&tid2,NULL,consumer, "2");
  pthread_create(&tid3,NULL,producer, "1");
  pthread_create(&tid4,NULL,producer, "2");


  /* now wait for threads to exit */
  pthread_join(tid1,NULL);
  pthread_join(tid2,NULL);
  pthread_join(tid3,NULL);
  pthread_join(tid4,NULL);
}