// Matrix multiplication on the GPU

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main() {

	// Random initialization
	srand(time(NULL));

	// Allocate input matrix A, B, C in the host and the device
	// d_ refers to memory on the device
	// h_ refers to memory on the host

	float *d_A, *d_B, *d_C;
	float *h_A, *h_B, *h_C;

	int matrixSize = 1024; // square matrix matrixSize * matrixSize
	int numElements = matrixSize * matrixSize;

	cudaMallocHost(&h_A, numElements * sizeof(float));
	cudaMallocHost(&h_B, numElements * sizeof(float));
	cudaMallocHost(&h_C, numElements * sizeof(float));

	cudaMalloc(&d_A, numElements * sizeof(float));
	cudaMalloc(&d_B, numElements * sizeof(float));
	cudaMalloc(&d_C, numElements * sizeof(float));

	// Initialize the input matrix A and B with random values
	for(int i = 0; i < numElements; i++) {
		h_A[i] = rand()/(float)RAND_MAX;
		h_B[i] = rand()/(float)RAND_MAX;
	}

	// Transfer matrix A and B from the host to the device

	// Kernel call to compute the matrix multiplication on the device

	// Transfer matrix C from the device to the host

	// Verify the results

	// Release memory

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);

	return 0;
}
