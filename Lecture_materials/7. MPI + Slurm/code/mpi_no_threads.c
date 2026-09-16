#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int process_rank, number_processes, hostname_length;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &number_processes);
    MPI_Get_processor_name(hostname, &hostname_length);

    printf("Hello from process %d out of %d. Running on node %s\n", process_rank, number_processes, hostname);

    MPI_Finalize();

    return 0;
}