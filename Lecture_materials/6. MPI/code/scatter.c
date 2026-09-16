#include <stdio.h>
#include <mpi.h>
#define SIZE 4

void main(int argc, char *argv[])
{
   int numtasks, rank, sendcount, recvcount, source;
   float sendbuf[SIZE][SIZE] = {
     {1, 2, 3, 4},
     {5, 6, 7, 8},
     {9, 10, 11, 12},
     {13, 14, 15, 16}  };
   float recvbuf[SIZE];

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

   if (numtasks == SIZE) {
     // define source task and elements to send/receive, then perform collective scatter
     source = 0;
     sendcount = SIZE;
     recvcount = SIZE;
     MPI_Scatter(sendbuf,sendcount,MPI_FLOAT,recvbuf,recvcount,MPI_FLOAT,source,MPI_COMM_WORLD);

     printf("Rank = %d  Results: %.0f %.0f %.0f %.0f\n",rank,recvbuf[0], recvbuf[1],recvbuf[2],recvbuf[3]);
     }
   else
     printf("Nope, you must specify -np %d processors for this example to run. Terminating.\n",SIZE);

   MPI_Finalize();
}
