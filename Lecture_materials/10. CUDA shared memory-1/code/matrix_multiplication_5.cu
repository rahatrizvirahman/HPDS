// Matrix multiplication on the GPU

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matrixMultiplicationGPU (float *A, float *B, float *C, int matrixSize) {

	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row    = blockIdx.y * blockDim.y + threadIdx.y;

	if(row < matrixSize && column < matrixSize) {
		
	}
}

void matrixMultiplicationCPU (float *A, float *B, float *C, int matrixSize) {

	for(int i = 0; i < matrixSize; i++) {
		for(int j = 0; j < matrixSize; j++) {
			float sum = 0;
			for(int k = 0; k < matrixSize; k++) {
				//sum += A[i][k] * B[k][j]; (2D mapping)
				sum += A[i * matrixSize + k] * B[k * matrixSize + j];
			}
			//C[i][j] = sum; (2D mapping)
			C[i * matrixSize + j] = sum;
		}
	}
}

int main() {

	// Random initialization
	srand(time(NULL));

	// Allocate input matrix A, B, C in the host and the device
	// d_ refers to memory on the device
	// h_ refers to memory on the host

	float *d_A, *d_B, *d_C;
	float *h_A, *h_B, *h_C, *h_C_CPU;

	int matrixSize = 1024; // square matrix matrixSize * matrixSize
	int numElements = matrixSize * matrixSize;

	cudaMallocHost(&h_A, numElements * sizeof(float));
	cudaMallocHost(&h_B, numElements * sizeof(float));
	cudaMallocHost(&h_C, numElements * sizeof(float));
	cudaMallocHost(&h_C_CPU, numElements * sizeof(float));

	cudaMalloc(&d_A, numElements * sizeof(float));
	cudaMalloc(&d_B, numElements * sizeof(float));
	cudaMalloc(&d_C, numElements * sizeof(float));

	// Initialize the input matrix A and B with random values
	for(int i = 0; i < numElements; i++) {
		h_A[i] = rand()/(float)RAND_MAX;
		h_B[i] = rand()/(float)RAND_MAX;
	}

	// Transfer matrix A and B from the host to the device

	cudaMemcpy(d_A, h_A, numElements * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, numElements * sizeof(float), cudaMemcpyHostToDevice);

	// Kernel call to compute the matrix multiplication on the device

	int numberofthreadsperDim = 16; // 2D: 16x16 = 256 threads per block
	int gridDim = (matrixSize + numberofthreadsperDim - 1 ) / numberofthreadsperDim;

    dim3 blockSize (numberofthreadsperDim, numberofthreadsperDim);
    dim3 gridSize (gridDim, gridDim);

	matrixMultiplicationGPU <<< gridSize, blockSize >>> (d_A, d_B, d_C, matrixSize);

	// Transfer matrix C from the device to the host

	cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);

	// Check error in CUDA

	cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {        
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	// Verify the results

	matrixMultiplicationCPU(h_A, h_B, h_C_CPU, matrixSize);

	for(int i = 0; i < numElements; i++) {
		if(fabs(h_C[i] - h_C_CPU[i]) > 1e-3) {
			fprintf(stderr, "Result verification failed at element %d, %f vs %f!\n", i, h_C[i], h_C_CPU[i]);
			exit(EXIT_FAILURE);
		}
	}

	// Release memory

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);
	cudaFreeHost(h_C_CPU);

	return 0;
}
