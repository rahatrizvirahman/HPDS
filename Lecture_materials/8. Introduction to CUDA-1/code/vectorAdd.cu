#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *A, float *B, float *C, int numElements)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < numElements) {
        C[tid] = A[tid] + B[tid];
    }
}

int main(int argc, char* argv[])
{
    int numElements = 50000;

    // Allocate host memory
    float *h_A = (float *)malloc(numElements * sizeof(float));
    float *h_B = (float *)malloc(numElements * sizeof(float));
    float *h_C = (float *)malloc(numElements * sizeof(float));

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vectors
    float *d_A, *d_B, *d_C;
    
    cudaMalloc(&d_A, numElements * sizeof(float));
    cudaMalloc(&d_B, numElements * sizeof(float));
    cudaMalloc(&d_C, numElements * sizeof(float));

    // Copy the host input vectors A and B from the host memory to the device memory
    cudaMemcpy(d_A, h_A, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, numElements * sizeof(float), cudaMemcpyHostToDevice);

    // Configure the blocks and grid sizes assuming a 1D thread block of 256 threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("CUDA kernel launch with %d 1D blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    // Launch the Vector Add CUDA Kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    // Copy the device result vector in device memory to the host result vector
    cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaError_t cudaError = cudaGetLastError();
  
    if(cudaError != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; i++) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
        
    printf("Sum of the vectors was OK\n");

    // Free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}