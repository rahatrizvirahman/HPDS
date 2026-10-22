#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matrixMul(float *A, float *B, float *C, int width)
{
	int column = ( blockDim.x * blockIdx.x ) + threadIdx.x;
	int row    = ( blockDim.y * blockIdx.y ) + threadIdx.y;

	if (row < width && column < width)
	{
		float sum = 0;

		for(int k = 0; k < width; k++)
			sum += A[row * width + k] * B[k * width + column];

		C[row*width + column] = sum;
	}
}

void MatrixMultiplicationHost(float *A, float *B, float *C, int width)
{
	for (int i = 0; i < width; i++)
		for (int j = 0; j < width; j++)
		{
			float sum = 0;

			for (int k = 0; k < width; k++)
				sum += A[i * width + k] * B[k * width + j];

			C[i * width + j] = sum;
		}
}

int main(int argc, char* argv[])
{
	int matrixSize = 1024; // square matrix matrixSize * matrixSize
	int numElements = matrixSize * matrixSize;

	// Allocate host memory
	float *h_A = (float *)malloc(numElements * sizeof(float));
	float *h_B = (float *)malloc(numElements * sizeof(float));
	float *h_C = (float *)malloc(numElements * sizeof(float));
	float *h_C_CPUres = (float *)malloc(numElements * sizeof(float));

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

	int threadsPerBlockDim = 16;
	int gridDimSize = (matrixSize + threadsPerBlockDim - 1) / threadsPerBlockDim;

	dim3 blockSize(threadsPerBlockDim, threadsPerBlockDim);
	dim3 gridSize (gridDimSize, gridDimSize);

	printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n", gridDimSize, gridDimSize, threadsPerBlockDim, threadsPerBlockDim);

	cudaEventRecord(start);

	matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, matrixSize);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU time to multiply matrixes %f ms\n", milliseconds);

	// Copy the device result matrix in device memory to the host result matrix
	cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);

	cudaError_t cudaError = cudaGetLastError();

	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
		exit(EXIT_FAILURE);
	}

	// Compute CPU time
	cudaEventRecord(start);

	MatrixMultiplicationHost(h_A, h_B, h_C_CPUres, matrixSize);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("CPU time to multiply matrixes %f ms\n", milliseconds);

	// Verify that the result matrix is correct
	for (int i = 0; i < numElements; i++)
		if (fabs(h_C[i] - h_C_CPUres[i]) > 1e-3)
		{
			fprintf(stderr, "Result verification failed at element %d, %f vs %f!\n", i, h_C[i], h_C_CPUres[i]);
			exit(EXIT_FAILURE);
		}

	printf("Multiplication of the matrixes was OK\n");

	// Free device global memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_C_CPUres);

	return 0;
}
