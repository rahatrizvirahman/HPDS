#include <stdio.h>
#include <mpi.h>

void main (int argc, char * argv[])
{
  int rank, size;

  /* start MPI */
  MPI_Init (&argc, &argv);	
  /* get current process id */
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);	
  /* get number of processes */
  MPI_Comm_size (MPI_COMM_WORLD, &size);	
  
  printf("Hello world from process %d of %d\n", rank, size);
  
  MPI_Finalize();
}