CXX = g++
NVCC ?= nvcc
MPICXX ?= mpicxx
CXXFLAGS ?= -O2 -std=c++11 -ffp-contract=off
NVCCFLAGS ?= -O2 -std=c++11 --fmad=false
CPPFLAGS ?= -I libarff
ARFF_SOURCES := $(wildcard libarff/*.cpp)
ARFF_HEADERS := $(wildcard libarff/*.h)

# MPI_HOME comes from the loaded Open MPI module. mpi_cuda is not part of "all".
MPI_HOME ?=

.PHONY: all cpu clean
all: serial cuda
cpu: serial threaded openmp mpi

serial: serial.cpp knn_io.h $(ARFF_SOURCES) $(ARFF_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) serial.cpp $(ARFF_SOURCES) -o $@
cuda: cuda.cu knn_cuda.cuh knn_io.h $(ARFF_SOURCES) $(ARFF_HEADERS)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) cuda.cu $(ARFF_SOURCES) -o $@
mpi_cuda: mpi_cuda.cu knn_cuda.cuh knn_io.h $(ARFF_SOURCES) $(ARFF_HEADERS)
	@test -n "$(MPI_HOME)" || { echo 'Load an Open MPI module so MPI_HOME is set' >&2; exit 1; }
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -I$(MPI_HOME)/include mpi_cuda.cu $(ARFF_SOURCES) -L$(MPI_HOME)/lib -lmpi -Xlinker -rpath -Xlinker $(MPI_HOME)/lib -o $@
threaded: threaded.cpp knn_io.h $(ARFF_SOURCES) $(ARFF_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread threaded.cpp $(ARFF_SOURCES) -o $@
openmp: openmp.cpp knn_io.h $(ARFF_SOURCES) $(ARFF_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fopenmp openmp.cpp $(ARFF_SOURCES) -o $@
mpi: mpi.cpp knn_io.h $(ARFF_SOURCES) $(ARFF_HEADERS)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) mpi.cpp $(ARFF_SOURCES) -o $@
clean:
	rm -f serial cuda mpi_cuda threaded openmp mpi
