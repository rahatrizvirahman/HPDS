#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduce_atomic(int *result, int *array, int numElements, int numberThreads)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < numberThreads)
    {
    	int localSum = 0;

    	for(int i = tid; i < numElements; i += numberThreads)
    		localSum += array[i];

    	atomicAdd(result, localSum);
    }
}

int main(int argc, char* argv[])
{
    int numElements = 1e6;

    // Allocate host memory
    int *h_array  = (int *)malloc(numElements * sizeof(int));
    int *h_result = (int *)malloc(sizeof(int));

    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

    // Initialize the host input vectors
    for (int i = 0; i < numElements; i++)
        h_array[i] = (i+1);

    // Allocate the device input vector
    int *d_array, *d_result;
    cudaMalloc(&d_array, numElements * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    // Copy the host input vector
    cudaMemcpy(d_array, h_array, numElements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(int));

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = 256;
    int numElementsPerThread = (numElements + (threadsPerBlock*blocksPerGrid) - 1) / (threadsPerBlock*blocksPerGrid);

    printf("%d blocks of %d threads. Each threads sums %d elements\n", blocksPerGrid, threadsPerBlock, numElementsPerThread);

    cudaEventRecord(start);

    reduce_atomic<<<blocksPerGrid, threadsPerBlock>>>(d_result, d_array, numElements, blocksPerGrid * threadsPerBlock);

    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU time %f ms\n", milliseconds);

    // Copy the result
    cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaError_t cudaError = cudaGetLastError();

    if(cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(start);

    int CPU_result = 0;

    for (int i = 0; i < numElements; i++)
    	CPU_result += h_array[i];

    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("CPU time %f ms\n", milliseconds);

    printf("GPU result %d, CPU result %d, %s!\n", *h_result, CPU_result, *h_result == CPU_result ? "CORRECT" : "ERROR" );

    // Free device global memory
    cudaFree(d_array);
    cudaFree(d_result);

    // Free host memory
    free(h_result);

    return 0;
}

