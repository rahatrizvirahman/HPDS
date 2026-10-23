#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#include <chrono>
#include "knn_io.h"
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include "mpi.h"

// Calculates the distance between two instances
float distance(float* instance_A, float* instance_B, int num_attributes) {
    float sum = 0;
    
    for (int i = 0; i < num_attributes-1; i++) {
        float diff = instance_A[i] - instance_B[i];
        sum += diff*diff;
    }
    
    return sqrt(sum);
}

// Implements a MPI kNN where for each candidate query an in-place priority queue is maintained to identify the nearest neighbors
int* KNN(ArffData* train, ArffData* test, int k, int mpi_rank, int mpi_num_processes) {

    int num_classes         = train->num_classes();
    int num_attributes      = train->num_attributes();
    int train_num_instances = train->num_instances();
    int test_num_instances  = test->num_instances();

    // Only rank 0 needs the full array in the end; every rank allocates it
    // anyway so the return type matches the other versions.
    int* predictions = (int*)calloc(test_num_instances, sizeof(int));

    // Each rank already parsed its own copy of the data in main(), so nothing
    // needs to be broadcast here.
    float* train_matrix = train->get_dataset_matrix();
    float* test_matrix  = test->get_dataset_matrix();

    // Full partition table on every rank, not just its own slice, since
    // MPI_Gatherv below needs every rank's count and offset.
    int* counts = (int*) malloc(mpi_num_processes * sizeof(int));
    int* displs = (int*) malloc(mpi_num_processes * sizeof(int));

    for(int r = 0; r < mpi_num_processes; r++) {
        int r_start = (int)(((long) r      * test_num_instances) / mpi_num_processes);
        int r_end   = (int)(((long)(r + 1) * test_num_instances) / mpi_num_processes);
        displs[r] = r_start;
        counts[r] = r_end - r_start;
    }

    int my_start = displs[mpi_rank];
    int my_count = counts[mpi_rank];

    // element i here is global test instance my_start + i
    int* local_predictions = (int*) calloc(my_count > 0 ? my_count : 1, sizeof(int));

    float* candidates = (float*) calloc(k*2, sizeof(float));
    for(int i = 0; i < 2*k; i++){ candidates[i] = FLT_MAX; }
    int* classCounts = (int*) calloc(num_classes, sizeof(int));

    for(int i = 0; i < my_count; i++) {
        int queryIndex = my_start + i;

        for(int keyIndex = 0; keyIndex < train_num_instances; keyIndex++) {

            float dist = distance(&test_matrix[queryIndex*num_attributes], &train_matrix[keyIndex*num_attributes], num_attributes);

            for(int c = 0; c < k; c++){
                if(dist < candidates[2*c]) {
                    for(int x = k-2; x >= c; x--) {
                        candidates[2*x+2] = candidates[2*x];
                        candidates[2*x+3] = candidates[2*x+1];
                    }
                    candidates[2*c] = dist;
                    candidates[2*c+1] = train_matrix[keyIndex*num_attributes + num_attributes - 1]; // class value
                    break;
                }
            }
        }

        for(int j = 0; j < k; j++) {
            classCounts[(int)candidates[2*j+1]] += 1;
        }

        int max_value = -1;
        int max_class = 0;
        for(int j = 0; j < num_classes; j++) {
            if(classCounts[j] > max_value) {
                max_value = classCounts[j];
                max_class = j;
            }
        }

        local_predictions[i] = max_class;

        for(int j = 0; j < 2*k; j++){ candidates[j] = FLT_MAX; }
        memset(classCounts, 0, num_classes * sizeof(int));
    }

    // Gatherv (not Gather) since blocks can differ by one when the test set
    // doesn't divide evenly across ranks
    MPI_Gatherv(local_predictions, my_count, MPI_INT,
                predictions, counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    free(local_predictions);
    free(candidates);
    free(classCounts);
    free(counts);
    free(displs);
    free(train_matrix);
    free(test_matrix);

    return predictions;
}

static int run(int argc, char *argv[], int mpi_rank, int mpi_num_processes)
{
    if(argc != 4 && argc != 5)
        throw std::runtime_error("Usage: ./mpi train.arff test.arff k [predictions.txt]");

    // k value for the k-nearest neighbors
    int k = positive_integer(argv[3]);

    // Every rank parses its own copy, before the timer starts
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();
    validate_datasets(train, test, k);

    // Sync ranks before starting the clock so an early finisher doesn't
    // count its wait inside MPI_Gatherv as compute time
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::steady_clock::now();
    int* predictions = KNN(train, test, k, mpi_rank, mpi_num_processes);
    auto end = std::chrono::steady_clock::now();

    // Report the slowest rank, as for the MPI+CUDA program
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double max_time_ms = 0;
    MPI_Reduce(&time_ms, &max_time_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        int correct = 0;
        for (int row = 0; row < test->num_instances(); ++row)
            if (predictions[row] == test->get_instance(row)->get(test->num_attributes() - 1)->operator int32())
                ++correct;
        float accuracy = 100.0f * correct / test->num_instances();
        if (argc == 5) write_predictions(argv[4], predictions, test->num_instances());
        printf("The %d-NN classifier required %.6f ms with %d processes. Accuracy was %.2f%%\n", k, max_time_ms, mpi_num_processes, accuracy);
        printf("RESULT strategy=mpi workers=%d end_to_end_ms=%.6f accuracy=%.8f\n", mpi_num_processes, max_time_ms, accuracy);
    }

    free(predictions);
    return 0;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_num_processes);
    try {
        run(argc, argv, mpi_rank, mpi_num_processes);
    } catch (const std::exception& error) {
        // Abort so a failed rank cannot leave the others waiting in a collective
        fprintf(stderr, "rank %d: %s\n", mpi_rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return 0;
}
