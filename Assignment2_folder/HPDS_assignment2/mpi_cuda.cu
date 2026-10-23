#include <mpi.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "libarff/arff_parser.h"
#include "knn_io.h"
#include "knn_cuda.cuh"

// One MPI rank drives one GPU. Every rank keeps the full training set and classifies a contiguous
// range of test queries with the same kernels as the single-GPU program, so each query keeps the
// serial training scan order. Predictions are gathered to rank 0 through host memory.
static const int INFO_LENGTH = 400;

static int run(int argc_in, char** argv_in, int rank, int size) {
    const CudaOptions options = parse_cuda_options(argc_in, argv_in);
    const int argc = static_cast<int>(options.positional.size());
    char** argv = const_cast<char**>(options.positional.data());
    if (argc != 5 && argc != 6)
        throw std::runtime_error("Usage: ./mpi_cuda train.arff test.arff k threads_per_block [predictions.txt] [--strategy=global|shared] [--tile=rows]");
    const int k = positive_integer(argv[3]);
    const int threads = positive_integer(argv[4]);
    const bool use_shared = options.strategy == "shared";
    const int tile_rows = options.tile_rows;

    // Every rank parses its own copy before the timer starts.
    ArffParser train_parser(argv[1]), test_parser(argv[2]);
    ArffData* train = train_parser.parse();
    ArffData* test = test_parser.parse();
    validate_datasets(train, test, k);
    const int ntrain = train->num_instances(), ntest = test->num_instances();
    const int attributes = train->num_attributes(), classes = train->num_classes();

    // Ranks on the same node get consecutive local ranks. The device index is the local rank modulo the
    // number of GPUs visible to this task, which works whether a task sees every node GPU or only its own.
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);
    MPI_Comm_free(&node_comm);
    int leader = local_rank == 0 ? 1 : 0, nodes = 0;
    MPI_Allreduce(&leader, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 1) throw std::runtime_error("No allocated CUDA device is visible");
    const int device = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (threads > properties.maxThreadsPerBlock || threads > properties.maxThreadsDim[0])
        throw std::runtime_error("Block size exceeds device limit");
    const size_t shared_bytes = use_shared ? checked_product(checked_product(tile_rows, attributes), sizeof(float)) : 0;
    if (shared_bytes > properties.sharedMemPerBlock)
        throw std::runtime_error("Tile exceeds the shared memory available to one block");

    // Test hook. MPI_CUDA_FAIL_RANK makes one rank fail before the first collective so a test can confirm that
    // the whole job aborts instead of leaving the other ranks waiting.
    const char* fail_rank = std::getenv("MPI_CUDA_FAIL_RANK");
    if (fail_rank && std::atoi(fail_rank) == rank) throw std::runtime_error("Injected failure for abort test");

    // Every rank must own a distinct physical GPU. A repeated UUID means ranks share a device.
    std::vector<unsigned char> uuids(static_cast<size_t>(size) * sizeof(properties.uuid.bytes));
    MPI_Allgather(properties.uuid.bytes, sizeof(properties.uuid.bytes), MPI_BYTE,
                  uuids.data(), sizeof(properties.uuid.bytes), MPI_BYTE, MPI_COMM_WORLD);
    for (int a = 0; a < size; ++a)
        for (int b = a + 1; b < size; ++b)
            if (std::memcmp(&uuids[a * sizeof(properties.uuid.bytes)], &uuids[b * sizeof(properties.uuid.bytes)],
                            sizeof(properties.uuid.bytes)) == 0)
                throw std::runtime_error("Two ranks were assigned the same GPU. Request one GPU per rank");
    char pci[32] = "unknown";
    cudaDeviceGetPCIBusId(pci, sizeof(pci), device);
    char host[MPI_MAX_PROCESSOR_NAME] = "";
    int host_length = 0;
    MPI_Get_processor_name(host, &host_length);

    // Contiguous query ranges as in the CPU MPI program. Remainders and empty ranges are valid.
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        displs[r] = static_cast<int>(static_cast<long>(r) * ntest / size);
        counts[r] = static_cast<int>(static_cast<long>(r + 1) * ntest / size) - displs[r];
    }
    const int my_start = displs[rank], my_count = counts[rank];
    const size_t blocks = (static_cast<size_t>(my_count) + threads - 1) / threads;
    if (blocks > static_cast<size_t>(properties.maxGridSize[0])) throw std::runtime_error("Grid size exceeds device limit");

    const size_t train_bytes = checked_product(checked_product(ntrain, attributes), sizeof(float));
    const size_t test_bytes = checked_product(checked_product(my_count, attributes), sizeof(float));
    const size_t candidate_count = checked_product(my_count, k);
    const size_t candidate_bytes = checked_product(candidate_count, sizeof(float));
    const size_t label_bytes = checked_product(candidate_count, sizeof(int));
    const size_t vote_bytes = checked_product(checked_product(my_count, classes), sizeof(int));
    const size_t prediction_bytes = checked_product(my_count, sizeof(int));

    std::vector<int> local_predictions(my_count > 0 ? my_count : 1);
    std::vector<int> predictions(rank == 0 ? ntest : 1);
    double kernel_ms = 0;

    // The timed region follows a barrier and covers flattening, transfers, the kernel, the gather and cleanup.
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::steady_clock::now();
    float* host_train = train->get_dataset_matrix();
    float* host_test = test->get_dataset_matrix();
    if (!host_train || !host_test) throw std::runtime_error("Host matrix allocation failed");
    if (my_count > 0) {
        float *device_train = NULL, *device_test = NULL, *distances = NULL;
        int *labels = NULL, *votes = NULL, *device_predictions = NULL;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_train), train_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_test), test_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&distances), candidate_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&labels), label_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&votes), vote_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_predictions), prediction_bytes));
        CUDA_CHECK(cudaMemcpy(device_train, host_train, train_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_test, host_test + static_cast<size_t>(my_start) * attributes, test_bytes,
                              cudaMemcpyHostToDevice));
        cudaEvent_t kernel_start, kernel_stop;
        CUDA_CHECK(cudaEventCreate(&kernel_start));
        CUDA_CHECK(cudaEventCreate(&kernel_stop));
        CUDA_CHECK(cudaEventRecord(kernel_start));
        if (use_shared)
            knn_shared<<<static_cast<unsigned int>(blocks), threads, shared_bytes>>>(
                device_train, device_test, ntrain, my_count, attributes, classes, k, tile_rows,
                distances, labels, votes, device_predictions);
        else
            knn_global<<<static_cast<unsigned int>(blocks), threads>>>(
                device_train, device_test, ntrain, my_count, attributes, classes, k,
                distances, labels, votes, device_predictions);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop));
        float elapsed = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, kernel_start, kernel_stop));
        kernel_ms = elapsed;
        CUDA_CHECK(cudaMemcpy(local_predictions.data(), device_predictions, prediction_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventDestroy(kernel_start));
        CUDA_CHECK(cudaEventDestroy(kernel_stop));
        CUDA_CHECK(cudaFree(device_train));
        CUDA_CHECK(cudaFree(device_test));
        CUDA_CHECK(cudaFree(distances));
        CUDA_CHECK(cudaFree(labels));
        CUDA_CHECK(cudaFree(votes));
        CUDA_CHECK(cudaFree(device_predictions));
    }
    // Gatherv because ranges differ by one query when ntest does not divide evenly.
    MPI_Gatherv(local_predictions.data(), my_count, MPI_INT, predictions.data(), counts.data(), displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);
    std::free(host_train);
    std::free(host_test);
    auto stop = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();

    // Per-rank evidence of GPU assignment and timing, collected after the timer stops.
    char info[INFO_LENGTH];
    std::snprintf(info, sizeof(info),
                  "RANK %d host=%s local_rank=%d visible_gpus=%d device=%d gpu=\"%s\" pci=%s queries=%d start=%d elapsed_ms=%.6f kernel_ms=%.6f",
                  rank, host, local_rank, device_count, device, properties.name, pci, my_count, my_start, elapsed_ms, kernel_ms);
    std::vector<char> all_info(rank == 0 ? static_cast<size_t>(size) * INFO_LENGTH : 1);
    MPI_Gather(info, INFO_LENGTH, MPI_CHAR, all_info.data(), INFO_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    double max_elapsed = 0, max_kernel = 0;
    MPI_Reduce(&elapsed_ms, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&kernel_ms, &max_kernel, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("MPI+CUDA strategy=%s ranks=%d nodes=%d threads_per_block=%d", options.strategy.c_str(), size, nodes, threads);
        if (use_shared) std::printf(" tile_rows=%d shared_bytes=%zu", tile_rows, shared_bytes);
        std::printf("\n");
        for (int r = 0; r < size; ++r) std::printf("%s\n", &all_info[static_cast<size_t>(r) * INFO_LENGTH]);
        int correct = 0;
        for (int row = 0; row < ntest; ++row) {
            if (predictions[row] < 0) throw std::runtime_error("Non-finite or excessive neighbor distances");
            if (predictions[row] == test->get_instance(row)->get(attributes - 1)->operator int32()) ++correct;
        }
        const float accuracy = 100.0f * correct / ntest;
        if (argc == 6) write_predictions(argv[5], predictions.data(), ntest);
        std::printf("The %d-NN classifier required %.6f ms on %d ranks. Accuracy was %.2f%%\n", k, max_elapsed, size, accuracy);
        std::printf("RESULT strategy=%s mpi_ranks=%d nodes=%d end_to_end_ms=%.6f kernel_ms=%.6f accuracy=%.8f\n",
                    options.strategy.c_str(), size, nodes, max_elapsed, max_kernel, accuracy);
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int status = 0;
    try {
        status = run(argc, argv, rank, size);
    } catch (const std::exception& error) {
        // A failed rank aborts the job so the other ranks cannot wait forever in a collective.
        std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
        std::fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
