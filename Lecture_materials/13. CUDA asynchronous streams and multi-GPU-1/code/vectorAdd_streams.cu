#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *A, float *B, float *C, int stream, int numberElementsPerStream)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = stream * numberElementsPerStream + tid;

    if (tid < numberElementsPerStream)
        for(int i = 0; i < 100; i++) // just to simulate a complex-enough kernel
            C[idx] = A[idx] + B[idx];
}

int main(int argc, char* argv[])
{
    int numElements = pow(2,24); //16M elements
    int numStreams = 4;

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

    cudaStream_t *streams = (cudaStream_t*) malloc (numStreams * sizeof(cudaStream_t));

    for (int i = 0; i < numStreams; i++)
        cudaStreamCreate(&streams[i]);

    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	cudaEventRecord(start);

	int threadsPerBlock = 256;
	int numberElementsPerStream = (numElements + numStreams - 1) / numStreams;
	int blocksPerGrid = (numberElementsPerStream + threadsPerBlock - 1) / threadsPerBlock;

	for (int i = 0; i < numStreams; i++)
	{
		// Copy the host input vectors A and B in host memory to the device input vectors in
		cudaMemcpyAsync(&d_A[i*numberElementsPerStream], &h_A[i*numberElementsPerStream], numberElementsPerStream * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(&d_B[i*numberElementsPerStream], &h_B[i*numberElementsPerStream], numberElementsPerStream * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

		vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_A, d_B, d_C, i, numberElementsPerStream);

		// Copy the device result vector in device memory to the host result vector
		cudaMemcpyAsync(&h_C[i*numberElementsPerStream], &d_C[i*numberElementsPerStream], numberElementsPerStream * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
	}

	cudaDeviceSynchronize();

    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU time %d streams %f ms\n", numStreams, milliseconds);

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

    free(streams);

    return 0;
}