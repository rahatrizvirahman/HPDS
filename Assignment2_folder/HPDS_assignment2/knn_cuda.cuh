#ifndef KNN_CUDA_CUH
#define KNN_CUDA_CUH

// Kernels and helpers shared by the single-GPU program (cuda.cu) and the MPI+CUDA program (mpi_cuda.cu).
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "knn_io.h"

static void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + " failed, " + cudaGetErrorString(status));
}
#define CUDA_CHECK(operation) check_cuda((operation), #operation)

static size_t checked_product(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::runtime_error("Allocation size overflow");
    return a * b;
}

// Both kernels share these helpers so the serial insertion and tie rules are defined once.
__device__ static void reset_query(float* distances, int* labels, int* counts, int k, int classes) {
    for (int c = 0; c < k; ++c) { distances[c] = FLT_MAX; labels[c] = -1; }
    for (int c = 0; c < classes; ++c) counts[c] = 0;
}

// Strict comparison keeps the earlier training row when distances are equal.
__device__ static void insert_candidate(float* distances, int* labels, int k, float distance, int label) {
    for (int c = 0; c < k; ++c) {
        if (distance < distances[c]) {
            for (int previous = k - 2; previous >= c; --previous) {
                distances[previous + 1] = distances[previous];
                labels[previous + 1] = labels[previous];
            }
            distances[c] = distance;
            labels[c] = label;
            break;
        }
    }
}

// The smallest class index wins a vote tie. Returns -1 when fewer than k finite candidates exist.
__device__ static int choose_class(const int* labels, int* counts, int k, int classes) {
    for (int c = 0; c < k; ++c) {
        if (labels[c] < 0 || labels[c] >= classes) return -1;
        ++counts[labels[c]];
    }
    int best_class = 0;
    for (int c = 1; c < classes; ++c)
        if (counts[c] > counts[best_class]) best_class = c;
    return best_class;
}

// Distance from one test row to one training row, accumulated in the serial feature order.
__device__ static float row_distance(const float* test_row, const float* train_row, int attributes) {
    float sum = 0.0f;
    for (int feature = 0; feature < attributes - 1; ++feature) {
        float diff = test_row[feature] - train_row[feature];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

// Lecture 8 indexing, with one independent query assigned to each thread.
__global__ void knn_global(const float* train, const float* test,
                           int ntrain, int ntest, int attributes, int classes, int k,
                           float* candidate_distance, int* candidate_label,
                           int* votes, int* predictions) {
    const size_t query = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (query >= static_cast<size_t>(ntest)) return;
    float* distances = candidate_distance + query * k;
    int* labels = candidate_label + query * k;
    int* counts = votes + query * classes;
    reset_query(distances, labels, counts, k, classes);
    const float* test_row = test + query * attributes;
    for (int row = 0; row < ntrain; ++row) {
        const float* train_row = train + static_cast<size_t>(row) * attributes;
        insert_candidate(distances, labels, k, row_distance(test_row, train_row, attributes),
                         static_cast<int>(train_row[attributes - 1]));
    }
    predictions[query] = choose_class(labels, counts, k, classes);
}

// Lecture 10 tiling. A block cooperatively loads tile_rows training rows into shared memory,
// every thread passes both barriers (including inactive threads of a partial query block),
// and active threads scan the tile in ascending row order before the next tile replaces it.
__global__ void knn_shared(const float* train, const float* test,
                           int ntrain, int ntest, int attributes, int classes, int k, int tile_rows,
                           float* candidate_distance, int* candidate_label,
                           int* votes, int* predictions) {
    extern __shared__ float tile[];
    const size_t query = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = query < static_cast<size_t>(ntest);
    float* distances = candidate_distance + query * k;
    int* labels = candidate_label + query * k;
    int* counts = votes + query * classes;
    const float* test_row = test + query * attributes;
    if (active) reset_query(distances, labels, counts, k, classes);
    for (int base = 0; base < ntrain; base += tile_rows) {
        const int rows = min(tile_rows, ntrain - base);
        const int values = rows * attributes;
        const float* source = train + static_cast<size_t>(base) * attributes;
        // Rows are contiguous in the flattened matrix, so adjacent threads load adjacent floats.
        for (int i = threadIdx.x; i < values; i += blockDim.x) tile[i] = source[i];
        __syncthreads();
        if (active) {
            for (int row = 0; row < rows; ++row) {
                const float* train_row = tile + row * attributes;
                insert_candidate(distances, labels, k, row_distance(test_row, train_row, attributes),
                                 static_cast<int>(train_row[attributes - 1]));
            }
        }
        __syncthreads();
    }
    if (active) predictions[query] = choose_class(labels, counts, k, classes);
}

// Optional flags may appear anywhere after the program name. The positional interface is unchanged.
struct CudaOptions {
    std::string strategy = "global";
    int tile_rows = 256;
    std::vector<char*> positional;
};

static CudaOptions parse_cuda_options(int argc, char** argv) {
    CudaOptions options;
    options.positional.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument.rfind("--strategy=", 0) == 0) options.strategy = argument.substr(11);
        else if (argument.rfind("--tile=", 0) == 0) options.tile_rows = positive_integer(argument.c_str() + 7);
        else options.positional.push_back(argv[i]);
    }
    if (options.strategy != "global" && options.strategy != "shared")
        throw std::runtime_error("Unknown strategy, expected global or shared");
    return options;
}

#endif
