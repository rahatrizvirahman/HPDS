all: reduction_atomic_1 reduction_atomic_2 reduction_atomic_3 reduction_atomic_4 reduction_shared_1 reduction_shared_2 thrust_sort thrust_sort_by_key thrust_reduce
reduction_atomic_1: reduction_atomic_1.cu
	nvcc reduction_atomic_1.cu -o reduction_atomic_1
reduction_atomic_2: reduction_atomic_2.cu
	nvcc reduction_atomic_2.cu -o reduction_atomic_2
reduction_atomic_3: reduction_atomic_3.cu
	nvcc reduction_atomic_3.cu -o reduction_atomic_3
reduction_atomic_4: reduction_atomic_4.cu
	nvcc reduction_atomic_4.cu -o reduction_atomic_4
reduction_shared_1: reduction_shared_1.cu
	nvcc reduction_shared_1.cu -o reduction_shared_1	
reduction_shared_2: reduction_shared_2.cu
	nvcc reduction_shared_2.cu -o reduction_shared_2	
thrust_sort: thrust_sort.cu
	nvcc thrust_sort.cu -o thrust_sort
thrust_sort_by_key: thrust_sort_by_key.cu
	nvcc thrust_sort_by_key.cu -o thrust_sort_by_key	
thrust_reduce: thrust_reduce.cu
	nvcc thrust_reduce.cu -o thrust_reduce	
clean:
	rm reduction_atomic_1 reduction_atomic_2 reduction_atomic_3 reduction_atomic_4 reduction_shared_1 reduction_shared_2 thrust_sort thrust_sort_by_key thrust_reduce