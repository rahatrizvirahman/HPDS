// Matrix multiplication on the GPU

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matrixMultiplicationTilesGPU (float *A, float *B, float *C, int matrixSize) {

	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row    = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int tiles_iterations = (matrixSize + TILE_SIZE - 1) / TILE_SIZE;

	if(row < matrixSize && column < matrixSize) {

		float sum = 0;

		for (int tile = 0; tile < tiles_iterations; tile++) {

			// A indexing [1+2+3]
			// 1. How many total elements there are above my row = matrixSize * row;
			// 2. How many total elements there are in the titles left of my current tile = tile * THREADS_DIM
			// 3. How many elements there are in the left within my tile = threadIdx.x
			// Memory access pattern is fully coalesced (consecutive threads in X dimension read consecutive memory positions in global memory)
			tile_A[threadIdx.y][threadIdx.x] = A[matrixSize * row + tile * TILE_SIZE + threadIdx.x];

			// B indexing [1+2+3]
			// 1. How many total elements there are in the tiles above my current tile = tile * THREADS_DIM * matrixSize
			// 2. How many total elements there are above within my tile = threadIdx.y * matrixSize			
			// 3. How many total elements there are left of my column = column
			tile_B[threadIdx.y][threadIdx.x] = B[tile * TILE_SIZE * matrixSize + threadIdx.y * matrixSize + column];

			__syncthreads();

			for(int k = 0; k < TILE_SIZE; k++) {
				sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
			}

			__syncthreads();
		}
		
		C[row * matrixSize + column] = sum;
	}
}

__global__ void matrixMultiplicationGPU (float *A, float *B, float *C, int matrixSize) {

	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row    = blockIdx.y * blockDim.y + threadIdx.y;

	if(row < matrixSize && column < matrixSize) {

		float sum = 0;

		for(int k = 0; k < matrixSize; k++) {
			//sum += A[row][k] + B[k][column];
			sum += A[row * matrixSize + k] * B[k * matrixSize + column];
		}
		
		C[row * matrixSize + column] = sum;
		// The previous line is equivalent to:  C[row][column] = sum;
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
			C[i * matrixSize + j] = sum;
			// The previous line is equivalent to:  C[i][j] = sum;
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

	// Create events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float runtime;

	cudaEventRecord(startEvent, 0);
	matrixMultiplicationCPU(h_A, h_B, h_C_CPU, matrixSize);
	cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&runtime, startEvent, stopEvent);

	printf("Naive matrix multiplication on CPU\t\t %10.4f ms\n", runtime);


	// Kernel call to compute the matrix multiplication on the device
	
	int numberofthreadsperDim = 16; // 2D: 16x16 = 256 threads per block
	int gridDim = (matrixSize + numberofthreadsperDim - 1 ) / numberofthreadsperDim;

    dim3 blockSize (numberofthreadsperDim, numberofthreadsperDim);
    dim3 gridSize (gridDim, gridDim);

    // Naive version
	cudaEventRecord(startEvent, 0);
	matrixMultiplicationGPU <<< gridSize, blockSize >>> (d_A, d_B, d_C, matrixSize);
	cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&runtime, startEvent, stopEvent);
    
    printf("Naive matrix multiplication on GPU\t\t %10.4f ms\n", runtime);

	// Tiles version
	cudaEventRecord(startEvent, 0);
	matrixMultiplicationTilesGPU <<< gridSize, blockSize >>> (d_A, d_B, d_C, matrixSize);
	cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&runtime, startEvent, stopEvent);
    
    printf("Tiled matrix multiplication on GPU\t\t %10.4f ms\n", runtime);

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