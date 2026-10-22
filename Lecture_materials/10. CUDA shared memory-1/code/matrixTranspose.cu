#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// naive transpose on the CPU
void transposeNaiveCPU(float *output, float *input, int matrixsize)
{
    for (int i = 0; i < matrixsize; i++)
        for (int j = 0; j < matrixsize; j++)
            output[j * matrixsize + i] = input[i * matrixsize + j];
}

// naive transpose on the GPU
// Simplest transpose; doesn't use shared memory.
// Global memory reads are coalesced but writes are not.
__global__ void transposeNaive(float *output, float *input, int matrixsize)
{    
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row    = blockIdx.y * blockDim.y + threadIdx.y;

    output[column*matrixsize + row] = input[row*matrixsize + column];
}

// shared memory transpose on the GPU
// Uses shared memory to achieve coalesing in both reads and writes
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
   
int main(int argc, char **argv)
{
    int matrixsize = 1024;

    // Allocate memory
    float *h_input = (float*)malloc(matrixsize*matrixsize*sizeof(float));
    float *h_output = (float*)malloc(matrixsize*matrixsize*sizeof(float));
    float *h_output_CPU = (float*)malloc(matrixsize*matrixsize*sizeof(float));

    float *d_input, *d_output;
    cudaMalloc(&d_input, matrixsize*matrixsize*sizeof(float));
    cudaMalloc(&d_output, matrixsize*matrixsize*sizeof(float));

    // Initialize array
    for(int i = 0; i < matrixsize*matrixsize; i++)
        h_input[i] = i;
   
    // Copy memory from host to device
    cudaMemcpy(d_input, h_input, matrixsize*matrixsize*sizeof(float), cudaMemcpyHostToDevice);

    // Create events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float runtime;

    // Transpose on the CPU
    cudaEventRecord(startEvent, 0);
    transposeNaiveCPU(h_output_CPU, h_input, matrixsize);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&runtime, startEvent, stopEvent);
    
    printf("Naive transpose on CPU\t\t %10.4f ms.\tBandwidth %10.2f GB/s\n", runtime, 2 * matrixsize * matrixsize * sizeof(float) * 1e-6 / runtime );

    // Naive transpose on the GPU
    int threadsPerBlock = 256;
    int numberBlocks = (matrixsize + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridNaiveTranspose(numberBlocks, matrixsize);

    cudaEventRecord(startEvent, 0);
    transposeNaive <<< gridNaiveTranspose, threadsPerBlock >>> (d_output, d_input, matrixsize);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&runtime, startEvent, stopEvent);

    cudaMemcpy(h_output, d_output, matrixsize*matrixsize*sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {        
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < matrixsize*matrixsize; i++)
        if(h_output[i] != h_output_CPU[i]) {
            printf("Naive transpose on GPU. Error in element %d. %f vs %f\n", i, h_output[i], h_output_CPU[i]);
            exit(0);
        }

    printf("Naive transpose on GPU\t\t %10.4f ms.\tBandwidth %10.2f GB/s\n", runtime, 2 * matrixsize * matrixsize * sizeof(float) * 1e-6 / runtime );
    
    // Transpose with shared memory on the GPU    
    dim3 blockSharedMemoryTranspose(TILE_SIZE, TILE_SIZE);
    dim3 gridSharedMemoryTranspose(matrixsize / TILE_SIZE, matrixsize / TILE_SIZE); // Assumes divisibility of matrix size / TILE_SIZE (16) (to make it easier to read the code)

    cudaEventRecord(startEvent, 0);
    transposeSharedMemoryGPU <<< gridSharedMemoryTranspose, blockSharedMemoryTranspose >>> (d_output, d_input, matrixsize);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&runtime, startEvent, stopEvent);

    cudaMemcpy(h_output, d_output, matrixsize*matrixsize*sizeof(float), cudaMemcpyDeviceToHost);

    error = cudaGetLastError();

    if (error != cudaSuccess)
    {        
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < matrixsize*matrixsize; i++)
        if(h_output[i] != h_output_CPU[i]) {
            printf("Shared memory transpose on GPU. Error in element %d. %f vs %f\n", i, h_output[i], h_output_CPU[i]);
            exit(0);
        }

    printf("Shared memory transpose on GPU\t %10.4f ms.\tBandwidth %10.2f GB/s\n", runtime, 2 * matrixsize * matrixsize * sizeof(float) * 1e-6 / runtime );
    
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    
    cudaFree(d_output);
    cudaFree(d_input);    
    free(h_input);
    free(h_output);
    free(h_output_CPU);
}