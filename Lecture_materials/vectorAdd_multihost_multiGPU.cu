#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <mpi.h>

__global__ void vectorAdd(float *A, float *B, float *C, int numElements)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numElements)
        for (int i = 0; i < 100; i++) // simulate heavy compute
            C[tid] = A[tid] + B[tid];
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int numElementsTotal = 1 << 24; // 16M elements globally
    int numElementsPerHost = numElementsTotal / world_size;

    printf("Rank %d of %d: handling %d elements\n", world_rank, world_size, numElementsPerHost);

    // Allocate host memory
    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, numElementsPerHost * sizeof(float));
    cudaMallocHost(&h_B, numElementsPerHost * sizeof(float));
    cudaMallocHost(&h_C, numElementsPerHost * sizeof(float));

    // Initialize input vectors (each host handles its chunk)
    for (int i = 0; i < numElementsPerHost; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Get number of local GPUs
    int numberGPUs = 0;
    cudaGetDeviceCount(&numberGPUs);

    if (numberGPUs == 0) {
        printf("Rank %d: No GPUs detected\n", world_rank);
        MPI_Finalize();
        return -1;
    }

    printf("Rank %d: Found %d GPUs\n", world_rank, numberGPUs);

    float **d_A = (float**) malloc(numberGPUs * sizeof(float*));
    float **d_B = (float**) malloc(numberGPUs * sizeof(float*));
    float **d_C = (float**) malloc(numberGPUs * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*) malloc(numberGPUs * sizeof(cudaStream_t));

    int numElementsPerGPU = (numElementsPerHost + numberGPUs - 1) / numberGPUs;
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElementsPerGPU + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate GPU memory and streams
    for (int i = 0; i < numberGPUs; i++) {
        cudaSetDevice(i);
        cudaMalloc(&d_A[i], numElementsPerGPU * sizeof(float));
        cudaMalloc(&d_B[i], numElementsPerGPU * sizeof(float));
        cudaMalloc(&d_C[i], numElementsPerGPU * sizeof(float));
        cudaStreamCreate(&streams[i]);
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    // Launch GPU work asynchronously
    for (int i = 0; i < numberGPUs; i++) {
        cudaSetDevice(i);
        int offset = i * numElementsPerGPU;
        int count = (offset + numElementsPerGPU <= numElementsPerHost)
                        ? numElementsPerGPU
                        : numElementsPerHost - offset;

        cudaMemcpyAsync(d_A[i], &h_A[offset], count * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_B[i], &h_B[offset], count * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_A[i], d_B[i], d_C[i], count);

        cudaMemcpyAsync(&h_C[offset], d_C[i], count * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    // Wait for all GPUs to finish
    for (int i = 0; i < numberGPUs; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
    printf("Rank %d completed local work in %llu ms\n", world_rank, (long long unsigned int) diff);

    // Gather all results on rank 0 using CUDA-aware MPI
    if (world_rank == 0) {
        float *globalC;
        cudaMallocHost(&globalC, numElementsTotal * sizeof(float));

        MPI_Gather(h_C, numElementsPerHost, MPI_FLOAT,
                   globalC, numElementsPerHost, MPI_FLOAT,
                   0, MPI_COMM_WORLD);

        // Verify results
        for (int i = 0; i < numElementsTotal; i++) {
            float expected = h_A[i % numElementsPerHost] + h_B[i % numElementsPerHost];
            if (fabs(globalC[i] - expected) > 1e-4) {
                fprintf(stderr, "Verification failed at element %d\n", i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        printf("All ranks verified successfully.\n");
        cudaFreeHost(globalC);
    } else {
        MPI_Gather(h_C, numElementsPerHost, MPI_FLOAT,
                   NULL, 0, MPI_FLOAT,
                   0, MPI_COMM_WORLD);
    }

    // Cleanup
    for (int i = 0; i < numberGPUs; i++) {
        cudaSetDevice(i);
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
        cudaStreamDestroy(streams[i]);
    }

    free(d_A); free(d_B); free(d_C); free(streams);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);

    MPI_Finalize();
    return 0;
}
