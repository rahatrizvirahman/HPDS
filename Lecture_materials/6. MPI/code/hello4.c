#include <stdio.h>
#include <unistd.h>
#include <mpi.h>

#define BUFSIZE 1024
#define TAG 0

void main (int argc, char * argv[])
{
    int i, rank, size;
    char buff[BUFSIZE];
    
    MPI_Status stat;
    MPI_Init (&argc, &argv);	
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    if(rank == 0)
    {
         printf("Process %d reporting: we have %d processes\n", rank, size);

         for(i=1; i<size; i++)
         {
           sprintf(buff, "Hello %d! ", i);
        
           MPI_Send(buff, BUFSIZE, MPI_CHAR, i, TAG, MPI_COMM_WORLD);
         }
         
         for(i=1; i<size; i++)
         {
           MPI_Recv(buff, BUFSIZE, MPI_CHAR, i, TAG, MPI_COMM_WORLD, &stat);
        
           printf("Message from %d: %s\n", i, buff);
         }
    }
    else
    {
         MPI_Recv(buff, BUFSIZE, MPI_CHAR, 0, TAG, MPI_COMM_WORLD, &stat);

         char hostname[1024];
         hostname[1023] = '\0';
         gethostname(hostname, 1023);

        sleep(10);
        
         sprintf(buff, "Process %d on %s reporting for duty!", rank, hostname);
        
         MPI_Send(buff, BUFSIZE, MPI_CHAR, 0, TAG, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}
