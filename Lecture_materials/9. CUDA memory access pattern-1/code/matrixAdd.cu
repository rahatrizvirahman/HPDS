#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matrixAddv1(float *A, float *B, float *C, int matrixSize)
{
    int column = ( blockDim.x * blockIdx.x ) + threadIdx.x;
    int row    = ( blockDim.y * blockIdx.y ) + threadIdx.y;

    if (row < matrixSize && column < matrixSize)
        C[row*matrixSize + column] = A[row*matrixSize + column] + B[row*matrixSize + column];
}

__global__ void matrixAddv2v3(float *A, float *B, float *C, int numElements)
{
    int column = ( blockDim.x * blockIdx.x ) + threadIdx.x;
    int row    = ( blockDim.y * blockIdx.y ) + threadIdx.y;
    int tid    = ( blockDim.x * gridDim.x * row ) + column;

    if (tid < numElements)
        C[tid] = A[tid] + B[tid];
}

int main(int argc, char* argv[])
{
    int matrixSize = 4096; // square matrix matrixSize * matrixSize
    int numElements = matrixSize * matrixSize;

    // Allocate host memory
    float *h_A = (float *)malloc(numElements * sizeof(float));
    float *h_B = (float *)malloc(numElements * sizeof(float));
    float *h_C = (float *)malloc(numElements * sizeof(float));

    // Initialize the host input matrixs
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input matrix A
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, numElements * sizeof(float));
    cudaMalloc(&d_B, numElements * sizeof(float));
    cudaMalloc(&d_C, numElements * sizeof(float));

    // Copy the host input matrixs A and B in host memory to the device input matrixs in
    cudaMemcpy(d_A, h_A, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, numElements * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // Option 1: 2D grid of 2D thread blocks 16x16 (OK)
    {
    int threadsPerBlockDim = 16;
    int gridDimSize = (matrixSize + threadsPerBlockDim - 1) / threadsPerBlockDim;

    dim3 blockSize(threadsPerBlockDim, threadsPerBlockDim);
    dim3 gridSize (gridDimSize, gridDimSize);

    printf("v1 CUDA kernel launch with %dx%d blocks of %dx%d threads\n", gridDimSize, gridDimSize, threadsPerBlockDim, threadsPerBlockDim);

    cudaEventRecord(start);

    matrixAddv1<<<gridSize, blockSize>>>(d_A, d_B, d_C, matrixSize);

    cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("v1 GPU option time to sum the matrixes %f ms\n", milliseconds);

    // Copy the device result matrix in device memory to the host result matrix
    cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t cudaError = cudaGetLastError();

    if(cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

    // Verify that the result matrix is correct
    for (int i = 0; i < numElements; i++)
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "v1 Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }

    printf("v1 Sum of the matrixes was OK\n");
    }


    // Option 2: 1D grid of 1D thread blocks 1x256 (INEFFICIENT ON PURPOSE), multiple memory transactions!!
    {
	int threadsPerBlock = 256;
	int gridDim = (numElements + threadsPerBlock - 1) / threadsPerBlock; // the dimensionality per grid dimension cannot be larger than 65536 for GPUs using CC 2.0

	dim3 blocksize(1, threadsPerBlock);

	printf("v2 CUDA kernel launch with %d blocks of 1x%d threads\n", gridDim, threadsPerBlock);

    cudaEventRecord(start);

    matrixAddv2v3<<<gridDim, blocksize>>>(d_A, d_B, d_C, numElements);

    cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("v2 GPU option time to sum the matrixes %f ms\n", milliseconds);

    // Copy the device result matrix in device memory to the host result matrix
    cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t cudaError = cudaGetLastError();

    if(cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

    // Verify that the result matrix is correct
    for (int i = 0; i < numElements; i++)
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "v2 Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }

    printf("v2 Sum of the matrixes was OK\n");
    }

	// Option 3: 1D grid of 1D thread blocks (MOST EFFICIENT), smaller number of larger transactions
    {
	int threadsPerBlock = 256;
	int gridDim = (numElements + threadsPerBlock - 1) / threadsPerBlock; // the dimensionality per grid dimension cannot be larger than 65536 for GPUs using CC 2.0

	printf("v3 CUDA kernel launch with %d blocks of %dx1 threads\n", gridDim, threadsPerBlock);

    cudaEventRecord(start);

    matrixAddv2v3<<<gridDim, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("v3 GPU option time to sum the matrixes %f ms\n", milliseconds);

    // Copy the device result matrix in device memory to the host result matrix
    cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t cudaError = cudaGetLastError();

    if(cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

    // Verify that the result matrix is correct
    for (int i = 0; i < numElements; i++)
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "v3 Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }

    printf("v3 Sum of the matrixes was OK\n");
    }

    // Compute CPU time
    cudaEventRecord(start);

    for (int i = 0; i < numElements; i++)
		h_C[i] = h_A[i] + h_B[i];

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CPU time to sum the matrixes %f ms\n", milliseconds);

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