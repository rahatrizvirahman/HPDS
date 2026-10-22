#include <stdio.h>
#include <stdint.h>
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
	for (int i = 0; i < numElements; i++)
	{
		h_A[i] = rand()/(float)RAND_MAX;
		h_B[i] = rand()/(float)RAND_MAX;
	}

	int numberGPUs = 0;
	// Retrieve number of GPUs for this machine
	cudaGetDeviceCount(&numberGPUs);

	if(numberGPUs == 0) {
		printf("There are no GPUs detected\n");
		return -1;
	}

	float **d_A, **d_B, **d_C;

	// Allocate pointers for each of the memory GPUs
	for(int i = 0; i < numberGPUs; i++) {
		d_A = (float**) malloc (numberGPUs * sizeof(float*));
		d_B = (float**) malloc (numberGPUs * sizeof(float*));
		d_C = (float**) malloc (numberGPUs * sizeof(float*));
	}

	cudaStream_t *streams;
	streams = (cudaStream_t*) malloc (numberGPUs * sizeof(cudaStream_t));

	int numberElementsPerGPU = (numElements + numberGPUs - 1) / numberGPUs;

	int threadsPerBlock = 256;
	int blocksPerGrid = (numberElementsPerGPU + threadsPerBlock - 1) / threadsPerBlock;

	// Allocate memory for each GPU
	for(int i = 0; i < numberGPUs; i++) {

		cudaSetDevice(i);

		cudaMalloc(&d_A[i], numberElementsPerGPU * sizeof(float));
		cudaMalloc(&d_B[i], numberElementsPerGPU * sizeof(float));
		cudaMalloc(&d_C[i], numberElementsPerGPU * sizeof(float));

		cudaStreamCreate(&streams[i]);
	}

	struct timespec start, end;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);

	for(int i = 0; i < numberGPUs; i++) {

		cudaSetDevice(i);

		cudaMemcpyAsync(d_A[i], &h_A[i*numberElementsPerGPU], numberElementsPerGPU * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(d_B[i], &h_B[i*numberElementsPerGPU], numberElementsPerGPU * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

		vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_A[i], d_B[i], d_C[i], numberElementsPerGPU);

		cudaMemcpyAsync(&h_C[i*numberElementsPerGPU], d_C[i], numberElementsPerGPU * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
	}

	for(int i = 0; i < numberGPUs; i++) {
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

	printf("Completed %d elements in %llu ms using %d GPUs: %d elements per GPU\n", numElements, (long long unsigned int) diff, numberGPUs, numberElementsPerGPU);

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

	for(int i = 0; i < numberGPUs; i++) {
		cudaSetDevice(i);
		cudaFree(d_A[i]);
		cudaFree(d_B[i]);
		cudaFree(d_C[i]);
	}

	free(d_A);
	free(d_B);
	free(d_C);

	// Free host memory
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);

	return 0;
}

