#include <stdio.h>
#include <mpi.h>

void main(int argc, char *argv[])
{
   int numtasks, rank;

   int buffer = 0;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

   if (rank == 0) {
      buffer = 123456;
   }

   printf("Before Bcast. Process %d. Buffer %d\n", rank, buffer);
   
   MPI_Bcast(&buffer, 1, MPI_INT, 0, MPI_COMM_WORLD);
   
   MPI_Barrier(MPI_COMM_WORLD); // Barrier to help print messages before/after in order. No need for an actual barrier.

   printf("After Bcast. Process %d. Buffer %d\n", rank, buffer);

   MPI_Finalize();
}
