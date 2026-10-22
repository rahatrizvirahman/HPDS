all: matrixAdd matrixMul
matrixAdd: matrixAdd.cu
	nvcc matrixAdd.cu -o matrixAdd
matrixMul: matrixMul.cu
	nvcc matrixMul.cu -o matrixMul	
clean:
	rm matrixAdd matrixMul