#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "libarff/arff_parser.h"
#include "knn_io.h"
#include "knn_cuda.cuh"

int main(int raw_argc, char** raw_argv) try {
    const CudaOptions options = parse_cuda_options(raw_argc, raw_argv);
    const std::string strategy = options.strategy;
    const int tile_rows = options.tile_rows;
    const int argc = static_cast<int>(options.positional.size());
    char** argv = const_cast<char**>(options.positional.data());
    if (argc != 5 && argc != 6) {
        std::fprintf(stderr, "Usage: ./cuda train.arff test.arff k threads_per_block [predictions.txt] [--strategy=global|shared] [--tile=rows]\n");
        return 1;
    }
    const int k = positive_integer(argv[3]);
    const int threads = positive_integer(argv[4]);
    ArffParser train_parser(argv[1]), test_parser(argv[2]);
    ArffData* train = train_parser.parse();
    ArffData* test = test_parser.parse();
    validate_datasets(train, test, k);
    const int ntrain = train->num_instances(), ntest = test->num_instances();
    const int attributes = train->num_attributes(), classes = train->num_classes();

    // Device discovery is setup outside the classification timer.
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 1) throw std::runtime_error("No allocated CUDA device is visible");
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    if (threads > properties.maxThreadsPerBlock || threads > properties.maxThreadsDim[0])
        throw std::runtime_error("Block size exceeds device limit");
    const size_t blocks = (static_cast<size_t>(ntest) + threads - 1) / threads;
    if (blocks > static_cast<size_t>(properties.maxGridSize[0]))
        throw std::runtime_error("Grid size exceeds device limit");
    const bool use_shared = strategy == "shared";
    const size_t shared_bytes = use_shared ? checked_product(checked_product(tile_rows, attributes), sizeof(float)) : 0;
    if (shared_bytes > properties.sharedMemPerBlock)
        throw std::runtime_error("Tile exceeds the shared memory available to one block");
    std::printf("GPU %s, visible_device=0, strategy=%s, threads_per_block=%d", properties.name, strategy.c_str(), threads);
    if (use_shared) std::printf(", tile_rows=%d, shared_bytes=%zu", tile_rows, shared_bytes);
    std::printf("\n");

    const size_t train_bytes = checked_product(checked_product(ntrain, attributes), sizeof(float));
    const size_t test_bytes = checked_product(checked_product(ntest, attributes), sizeof(float));
    const size_t candidate_count = checked_product(ntest, k);
    const size_t candidate_bytes = checked_product(candidate_count, sizeof(float));
    const size_t label_bytes = checked_product(candidate_count, sizeof(int));
    const size_t vote_bytes = checked_product(checked_product(ntest, classes), sizeof(int));
    const size_t prediction_bytes = checked_product(ntest, sizeof(int));

    auto start = std::chrono::steady_clock::now();
    float* host_train = train->get_dataset_matrix();
    float* host_test = test->get_dataset_matrix();
    if (!host_train || !host_test) throw std::runtime_error("Host matrix allocation failed");
    std::vector<int> predictions(ntest);
    float *device_train = NULL, *device_test = NULL, *distances = NULL;
    int *labels = NULL, *votes = NULL, *device_predictions = NULL;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_train), train_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_test), test_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&distances), candidate_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&labels), label_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&votes), vote_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_predictions), prediction_bytes));
    CUDA_CHECK(cudaMemcpy(device_train, host_train, train_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_test, host_test, test_bytes, cudaMemcpyHostToDevice));
    cudaEvent_t kernel_start, kernel_stop;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventRecord(kernel_start));
    if (use_shared)
        knn_shared<<<static_cast<unsigned int>(blocks), threads, shared_bytes>>>(
            device_train, device_test, ntrain, ntest, attributes, classes, k, tile_rows,
            distances, labels, votes, device_predictions);
    else
        knn_global<<<static_cast<unsigned int>(blocks), threads>>>(
            device_train, device_test, ntrain, ntest, attributes, classes, k,
            distances, labels, votes, device_predictions);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));
    float kernel_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
    CUDA_CHECK(cudaMemcpy(predictions.data(), device_predictions, prediction_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaFree(device_train));
    CUDA_CHECK(cudaFree(device_test));
    CUDA_CHECK(cudaFree(distances));
    CUDA_CHECK(cudaFree(labels));
    CUDA_CHECK(cudaFree(votes));
    CUDA_CHECK(cudaFree(device_predictions));
    std::free(host_train);
    std::free(host_test);
    auto stop = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();

    int correct = 0;
    for (int row = 0; row < ntest; ++row) {
        if (predictions[row] < 0) throw std::runtime_error("Non-finite or excessive neighbor distances");
        if (predictions[row] == test->get_instance(row)->get(attributes - 1)->operator int32()) ++correct;
    }
    float accuracy = 100.0f * correct / ntest;
    if (argc == 6) write_predictions(argv[5], predictions.data(), ntest);
    std::printf("The %d-NN classifier required %.6f ms. Accuracy was %.2f%%\n", k, elapsed_ms, accuracy);
    std::printf("RESULT strategy=%s end_to_end_ms=%.6f kernel_ms=%.6f accuracy=%.8f\n",
                strategy.c_str(), elapsed_ms, kernel_ms, accuracy);
    return 0;
} catch (const std::exception& error) {
    // A failed run exits without emitting a RESULT record.
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
