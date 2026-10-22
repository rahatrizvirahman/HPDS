todo: vectorAdd_nostreams vectorAdd_streams vectorAdd_multigpu
vectorAdd_nostreams: vectorAdd_nostreams.cu
	nvcc --ptxas-options=-v -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 vectorAdd_nostreams.cu -o vectorAdd_nostreams
vectorAdd_streams: vectorAdd_streams.cu
	nvcc --ptxas-options=-v -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 vectorAdd_streams.cu -o vectorAdd_streams
vectorAdd_multigpu: vectorAdd_multigpu.cu
	nvcc --ptxas-options=-v -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 vectorAdd_multigpu.cu -o vectorAdd_multigpu	
clean:
	rm vectorAdd_nostreams vectorAdd_streams vectorAdd_multigpu