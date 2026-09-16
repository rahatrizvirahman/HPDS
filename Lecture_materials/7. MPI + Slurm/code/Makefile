all: mpi_no_threads mpi_openmp_threads matrixmultiplication_mpi matrixmultiplication_mpi_openmp
mpi_no_threads: mpi_no_threads.c
	mpicc mpi_no_threads.c -o mpi_no_threads
mpi_openmp_threads: mpi_openmp_threads.c
	mpicc -fopenmp mpi_openmp_threads.c -o mpi_openmp_threads
matrixmultiplication_mpi: matrixmultiplication_mpi.c
	mpicc -std=gnu11 matrixmultiplication_mpi.c -o matrixmultiplication_mpi	
matrixmultiplication_mpi_openmp: matrixmultiplication_mpi_openmp.c
	mpicc -fopenmp -std=gnu11 matrixmultiplication_mpi_openmp.c -o matrixmultiplication_mpi_openmp			
clean:
	rm mpi_no_threads mpi_openmp_threads matrixmultiplication_mpi matrixmultiplication_mpi_openmp