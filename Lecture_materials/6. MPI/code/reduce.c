#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

float *create_rand_nums(int num_elements) {
    float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
    int i;
    for (i = 0; i < num_elements; i++) {
    rand_nums[i] = (rand() / (float) RAND_MAX);
    }
    return rand_nums;
}

void main(int argc, char *argv[])
{
    int rank, ntasks, i;
    int num_elements_per_proc = 1000;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create a random array of elements on all processes.
    srand(time(NULL)*rank);   // Seed the random number generator to get different results each time for each processor
    
    float *rand_nums = create_rand_nums(num_elements_per_proc);

    // Sum the numbers locally
    float local_sum = 0;
    for (i = 0; i < num_elements_per_proc; i++) {
        local_sum += rand_nums[i];
    }

    // Print the random numbers on each process
    printf("Local sum for process %d %f\n", rank, local_sum);

    // Reduce all of the local sums into the global sum
    float global_sum;
    
    // See https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total sum = %f\n", global_sum);
    }

    free(rand_nums);

    MPI_Finalize();
}