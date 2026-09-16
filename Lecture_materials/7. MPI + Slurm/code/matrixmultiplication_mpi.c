#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <time.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {

    srand(time(NULL));
    
    int rank, numberProcesses;

    MPI_Init (&argc, &argv);	

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numberProcesses);

    int matrixsize = 256;

    double *full_matrix_A, *full_matrix_B, *full_matrix_C;
    double *part_matrix_A, *part_matrix_C;

    if(rank == 0) {

        full_matrix_A = (double*) malloc(matrixsize * matrixsize * sizeof(double));    
        full_matrix_B = (double*) malloc(matrixsize * matrixsize * sizeof(double));
        full_matrix_C = (double*) malloc(matrixsize * matrixsize * sizeof(double));

        for(int i = 0; i < matrixsize * matrixsize; i++) {
            full_matrix_A[i] = rand() / (double) RAND_MAX;
            full_matrix_B[i] = rand() / (double) RAND_MAX;            
        }

        part_matrix_A = (double*) malloc(matrixsize * matrixsize / numberProcesses * sizeof(double));    
        part_matrix_C = (double*) malloc(matrixsize * matrixsize / numberProcesses * sizeof(double));

    } else {
        part_matrix_A = (double*) malloc(matrixsize * matrixsize / numberProcesses * sizeof(double));    
        part_matrix_C = (double*) malloc(matrixsize * matrixsize / numberProcesses * sizeof(double));
        full_matrix_B = (double*) malloc(matrixsize * matrixsize * sizeof(double));
    }   

    struct timespec start, end;

    // Initialize time measurement
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    MPI_Scatter(full_matrix_A , matrixsize * matrixsize / numberProcesses, MPI_DOUBLE, part_matrix_A, matrixsize * matrixsize / numberProcesses, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(full_matrix_B, matrixsize * matrixsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for(int i = 0; i < matrixsize / numberProcesses; i++) {
        for(int j = 0; j < matrixsize; j++) {
            double tmp = 0;
            for(int k = 0; k < matrixsize; k++) {
                tmp += part_matrix_A[i * matrixsize + k] * full_matrix_B[k * matrixsize + j];
            }
            part_matrix_C[i*matrixsize + j] = tmp;
        }
    }

    MPI_Gather(part_matrix_C, matrixsize * matrixsize / numberProcesses, MPI_DOUBLE, full_matrix_C, matrixsize * matrixsize / numberProcesses, MPI_DOUBLE, 0 ,MPI_COMM_WORLD);

    // Stop time measurement
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t time_difference_mpi = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    // Verify if matrix multiplication with MPI matches the sequential one
    if(rank == 0) {
        double *full_matrix_C_comparable = (double*) malloc(matrixsize * matrixsize * sizeof(double));

        // Initialize time measurement
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);

        for(int i = 0; i < matrixsize; i++) {
            for(int j = 0; j < matrixsize; j++) {
                double tmp = 0;
                for(int k = 0; k < matrixsize; k++) {
                    tmp += full_matrix_A[i * matrixsize + k] * full_matrix_B[k * matrixsize + j];
                }
                full_matrix_C_comparable[i*matrixsize + j] = tmp;
            }
        }

        // Stop time measurement
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        uint64_t time_difference_sequential = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

        double error = 0;        
        for(int i = 0; i < matrixsize * matrixsize; i++) {
            error += fabs(full_matrix_C_comparable[i] - full_matrix_C[i]);
        }
        
        if(error == 0)
            printf("Success!!! Runtime %llu ms MPI and %llu ms sequential\n", (long long unsigned int) time_difference_mpi, (long long unsigned int) time_difference_sequential);
        else
            printf("Error in matrix differences %lf. You cry now\n", error);

        free(full_matrix_C_comparable);
    }

    // Free memory
    if(rank == 0) {
        free(full_matrix_A);
        free(full_matrix_B);
        free(full_matrix_C);
        free(part_matrix_A);
        free(part_matrix_C);
    } else {
        free(part_matrix_A);
        free(full_matrix_B);
        free(part_matrix_C);
    }

    MPI_Finalize();
}
