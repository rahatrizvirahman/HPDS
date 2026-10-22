CXX = g++
NVCC ?= nvcc
MPICXX ?= mpicxx
CXXFLAGS ?= -O2 -std=c++11 -ffp-contract=off
NVCCFLAGS ?= -O2 -std=c++11 --fmad=false
CPPFLAGS ?= -I libarff
ARFF_SOURCES := $(wildcard libarff/*.cpp)
ARFF_HEADERS := $(wildcard libarff/*.h)

.PHONY: all cpu clean
all: serial cuda
cpu: serial threaded openmp mpi

serial: serial.cpp knn_io.h $(ARFF_SOURCES) $(ARFF_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) serial.cpp $(ARFF_SOURCES) -o $@
cuda: cuda.cu knn_io.h $(ARFF_SOURCES) $(ARFF_HEADERS)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) cuda.cu $(ARFF_SOURCES) -o $@
threaded: threaded.cpp $(ARFF_SOURCES) $(ARFF_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread threaded.cpp $(ARFF_SOURCES) -o $@
openmp: openmp.cpp $(ARFF_SOURCES) $(ARFF_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fopenmp openmp.cpp $(ARFF_SOURCES) -o $@
mpi: mpi.cpp $(ARFF_SOURCES) $(ARFF_HEADERS)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) mpi.cpp $(ARFF_SOURCES) -o $@
clean:
	rm -f serial cuda threaded openmp mpi
