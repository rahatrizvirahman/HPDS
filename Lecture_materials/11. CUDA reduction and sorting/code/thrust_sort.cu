#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>

int main(int argc, char* argv[])
{
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	
    // generate 16M random numbers on the host
    thrust::host_vector<int> h_vec(1 << 24);
    
    thrust::generate(h_vec.begin(), h_vec.end(), rand);
    
    // transfer data to the device
    thrust::device_vector<int> d_vec = h_vec;
    
    cudaEventRecord(start);
    
    // sort data on the device
    thrust::sort(d_vec.begin(), d_vec.end());
    
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU time %f ms\n", milliseconds);
	
	// transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
	
	cudaEventRecord(start);
	
	std::sort(h_vec.begin(), h_vec.end());
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("CPU time %f ms\n", milliseconds);
    
    return 0;
}
