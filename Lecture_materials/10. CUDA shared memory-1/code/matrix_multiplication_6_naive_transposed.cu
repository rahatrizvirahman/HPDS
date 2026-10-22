// Matrix multiplication on the GPU

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void transposeSharedMemoryGPU(float *output, float *input, int matrixsize)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row    = blockIdx.y * blockDim.y + threadIdx.y;

    // Reading fully coalesced from global memory. Copying data into shared memory
    tile[threadIdx.y][threadIdx.x] = input[row*matrixsize + column];

    // Synchronize all threads within the block
    __syncthreads();

    // Swap row/col to write output memory
    column = blockIdx.y * blockDim.y + threadIdx.x; 
    row    = blockIdx.x * blockDim.x + threadIdx.y;

    // Writing fully coalesced to global memory.
    output[row*matrixsize + column] = tile[threadIdx.x][threadIdx.y];
}

__global__ void matrixMultiplicationGPUTransposed (float *A, float *B, float *C, int matrixSize) {

	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row    = blockIdx.y * blockDim.y + threadIdx.y;

	if(row < matrixSize && column < matrixSize) {

		float sum = 0;

		for(int k = 0; k < matrixSize; k++) {
			//sum += A[row][k] + B[column][k];
			sum += A[row * matrixSize + k] * B[column * matrixSize + k]; // pointers to matrix B are inverted to get fully coalesced memory accesses
		}
		
		//C[row][column] = sum;
		C[row * matrixSize + column] = sum;

		//printf("[%d][%d] %f\n", row, column, sum);
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
		
		//C[row][column] = sum;
		C[row * matrixSize + column] = sum;

		//printf("[%d][%d] %f\n", row, column, sum);
	}
}

void transposeMatrixCPU (float *input, float *output, int matrixSize) {
	
	for(int i = 0; i < matrixSize; i++) {
		for(int j = 0; j < matrixSize; j++) {
			output[j*matrixSize + i] = input[i*matrixSize + j];
		}
	}
}

void matrixMultiplicationCPUTransposed (float *A, float *B, float *C, int matrixSize) {

	for(int i = 0; i < matrixSize; i++) {
		for(int j = 0; j < matrixSize; j++) {
			float sum = 0;
			for(int k = 0; k < matrixSize; k++) {
				//sum += A[i][k] * B[k][j]; (2D mapping)
				sum += A[i * matrixSize + k] * B[j * matrixSize + k];
			}
			//C[i][j] = sum; (2D mapping)
			C[i * matrixSize + j] = sum;
		}
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

	float *d_A, *d_B, *d_B_transposed, *d_C;
	float *h_A, *h_B, *h_B_transposed, *h_C, *h_C_CPU;

	int matrixSize = 1024; // square matrix matrixSize * matrixSize
	int numElements = matrixSize * matrixSize;

	cudaMallocHost(&h_A, numElements * sizeof(float));
	cudaMallocHost(&h_B, numElements * sizeof(float));
	cudaMallocHost(&h_B_transposed, numElements * sizeof(float));
	cudaMallocHost(&h_C, numElements * sizeof(float));
	cudaMallocHost(&h_C_CPU, numElements * sizeof(float));

	cudaMalloc(&d_A, numElements * sizeof(float));
	cudaMalloc(&d_B, numElements * sizeof(float));
	cudaMalloc(&d_B_transposed, numElements * sizeof(float));
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

	// 1. Naive matrix multiplication on the CPU
	cudaEventRecord(startEvent, 0);
	matrixMultiplicationCPU(h_A, h_B, h_C_CPU, matrixSize);
	cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&runtime, startEvent, stopEvent);
    
    printf("Naive matrix multiplication on CPU\t\t %10.4f ms\n", runtime);

	// 2. Transpose matrix B and multiply with transposed matrix on the CPU
	cudaEventRecord(startEvent, 0);
	transposeMatrixCPU(h_B, h_B_transposed, matrixSize);
	matrixMultiplicationCPUTransposed(h_A, h_B_transposed, h_C_CPU, matrixSize);
	cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&runtime, startEvent, stopEvent);
    
    printf("Transposed matrix multiplication on CPU\t\t %10.4f ms\n", runtime);


	// 3. Naive matrix multiplication on the GPU

	// Kernel call to compute the matrix multiplication on the device

	int numberofthreadsperDim = 16; // 2D: 16x16 = 256 threads per block
	int gridDim = (matrixSize + numberofthreadsperDim - 1 ) / numberofthreadsperDim;

    dim3 blockSize (numberofthreadsperDim, numberofthreadsperDim);
    dim3 gridSize (gridDim, gridDim);

 	cudaEventRecord(startEvent, 0);
	matrixMultiplicationGPU <<< gridSize, blockSize >>> (d_A, d_B, d_C, matrixSize);
	cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&runtime, startEvent, stopEvent);
    
    printf("Naive matrix multiplication on GPU\t\t %10.4f ms\n", runtime);

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

	cudaMemset(d_C, 0, numElements * sizeof(float));

	// 4. Transpose matrix B and multiply with transposed matrix on the GPU

	// Transpose with shared memory on the GPU    
    dim3 blockSharedMemoryTranspose(TILE_SIZE, TILE_SIZE);
    dim3 gridSharedMemoryTranspose(matrixSize / TILE_SIZE, matrixSize / TILE_SIZE); // Assumes divisibility of matrix size / TILE_SIZE (16) (to make it easier to read the code)

    cudaEventRecord(startEvent, 0);
    transposeSharedMemoryGPU <<< gridSharedMemoryTranspose, blockSharedMemoryTranspose >>> (d_B_transposed, d_B, matrixSize);
	matrixMultiplicationGPUTransposed <<< gridSize, blockSize >>> (d_A, d_B_transposed, d_C, matrixSize);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&runtime, startEvent, stopEvent);
    
    printf("Transposed matrix multiplication on GPU\t\t %10.4f ms\n", runtime);

	// Transfer matrix C from the device to the host

	cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);

	// Check error in CUDA

	error = cudaGetLastError();

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
	cudaFree(d_B_transposed);
	cudaFree(d_C);

	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_B_transposed);
	cudaFreeHost(h_C);
	cudaFreeHost(h_C_CPU);

	return 0;
}
