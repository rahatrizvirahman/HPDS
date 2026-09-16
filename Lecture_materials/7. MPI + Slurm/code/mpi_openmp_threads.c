#include <stdio.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
    int process_rank, number_processes, hostname_length;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &number_processes);
    MPI_Get_processor_name(hostname, &hostname_length);

    #pragma omp parallel
    printf("Hello from process %d out of %d. Thread %d out of %d. Running on node %s\n", process_rank, number_processes, omp_get_thread_num(), omp_get_num_threads(), hostname);

    MPI_Finalize();

    return 0;
}