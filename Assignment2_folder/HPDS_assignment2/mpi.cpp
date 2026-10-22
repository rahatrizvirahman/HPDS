#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
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

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) { // for each instance compare the true class and predicted class    
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++) {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }
    
    return 100 * successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[])
{
    if(argc != 4)
    {
        printf("Usage: ./program datasets/train.arff datasets/test.arff k");
        exit(0);
    }

    // k value for the k-nearest neighbors
    int k = strtol(argv[3], NULL, 10);

    int mpi_rank, mpi_num_processes;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &mpi_num_processes);

    // Every rank parses its own copy, before the timer starts
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();
    
    struct timespec start, end;
    int* predictions = NULL;

    // Sync ranks before starting the clock so an early finisher doesn't
    // count its wait inside MPI_Gatherv as compute time
    MPI_Barrier(MPI_COMM_WORLD);

    // Initialize time measurement
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    predictions = KNN(train, test, k, mpi_rank, mpi_num_processes);
    
    // Stop time measurement
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    if (mpi_rank == 0) {
        // Compute the confusion matrix
        int* confusionMatrix = computeConfusionMatrix(predictions, test);
        // Calculate the accuracy
        float accuracy = computeAccuracy(confusionMatrix, test);

        uint64_t time_difference = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

        printf("The %i-NN classifier for %lu test instances and %lu train instances required %llu ms CPU time for MPI with %d processes. Accuracy was %.2f%%\n", k, test->num_instances(), train->num_instances(), (long long unsigned int) time_difference, mpi_num_processes, accuracy);

        free(confusionMatrix);
    }

    free(predictions);

    MPI_Finalize();
}
