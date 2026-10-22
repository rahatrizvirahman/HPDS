#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *A, float *B, float *C, int numElements)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < numElements)
        for(int i = 0; i < 100; i++) // just to simulate a complex-enough kernel
            C[tid] = A[tid] + B[tid];
}

int main(int argc, char* argv[])
{
    int numElements = pow(2,24); //16M elements

    // Allocate host memory
    float *h_A, *h_B, *h_C;

    cudaMallocHost(&h_A, numElements * sizeof(float));
    cudaMallocHost(&h_B, numElements * sizeof(float));
    cudaMallocHost(&h_C, numElements * sizeof(float));

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, numElements * sizeof(float));
    cudaMalloc(&d_B, numElements * sizeof(float));
    cudaMalloc(&d_C, numElements * sizeof(float));

    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	cudaEventRecord(start);

    // Copy the host input vectors A and B in host memory to the device input vectors in
    cudaMemcpy(d_A, h_A, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, numElements * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Copy the device result vector in device memory to the host result vector
    cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU time single stream %f ms\n", milliseconds);

    cudaError_t cudaError = cudaGetLastError();

    if(cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; i++)
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }

    printf("Sum of the vectors was OK\n");

    // Free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}