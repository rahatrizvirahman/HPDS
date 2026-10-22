#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <bits/stdc++.h>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

// Calculates the distance between two instances
float distance(float* instance_A, float* instance_B, int num_attributes) {
    float sum = 0;
    
    for (int i = 0; i < num_attributes-1; i++) {
        float diff = instance_A[i] - instance_B[i];
        sum += diff*diff;
    }
    
    return sqrt(sum);
}

// Implements a OpenMP kNN where for each candidate query an in-place priority queue is maintained to identify the nearest neighbors
int* KNN(ArffData* train, ArffData* test, int k) {    

    int num_classes         = train->num_classes();
    int num_attributes      = train->num_attributes();
    int train_num_instances = train->num_instances();
    int test_num_instances  = test->num_instances();

    int* predictions = (int*)calloc(test_num_instances, sizeof(int));

    // get_dataset_matrix() rebuilds the matrix each call, so build it once here
    float* train_matrix = train->get_dataset_matrix();
    float* test_matrix  = test->get_dataset_matrix();

    // "omp parallel" + "omp for" instead of the fused "omp parallel for", so
    // candidates/classCounts can be allocated once per thread below, not once
    // per iteration. Declaring them here (rather than via private()) is what
    // makes them private, since private()/firstprivate() only copy the
    // pointer, not the buffer it points to.
    #pragma omp parallel
    {
        float* candidates = (float*) calloc(k*2, sizeof(float));
        for(int i = 0; i < 2*k; i++){ candidates[i] = FLT_MAX; }
        int* classCounts = (int*) calloc(num_classes, sizeof(int));

        #pragma omp for
        for(int queryIndex = 0; queryIndex < test_num_instances; queryIndex++) {
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

            for(int i = 0; i < k; i++) {
                classCounts[(int)candidates[2*i+1]] += 1;
            }

            int max_value = -1;
            int max_class = 0;
            for(int i = 0; i < num_classes; i++) {
                if(classCounts[i] > max_value) {
                    max_value = classCounts[i];
                    max_class = i;
                }
            }

            predictions[queryIndex] = max_class;

            for(int i = 0; i < 2*k; i++){ candidates[i] = FLT_MAX; }
            memset(classCounts, 0, num_classes * sizeof(int));
        }
        free(candidates);
        free(classCounts);
    }

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
    if(argc != 5)
    {    
        printf("Usage: ./program datasets/train.arff datasets/test.arff k num_threads");
        exit(0);
    }

    // k value for the k-nearest neighbors
    int k = strtol(argv[3], NULL, 10);
    int num_threads = strtol(argv[4], NULL, 10);

    omp_set_num_threads(num_threads);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();
    
    struct timespec start, end;
    int* predictions = NULL;
    
    // Initialize time measurement
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    predictions = KNN(train, test, k);
    
    // Stop time measurement
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(predictions, test);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, test);

    uint64_t time_difference = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("The %i-NN classifier for %lu test instances and %lu train instances required %llu ms CPU time for OpenMP with %d threads. Accuracy was %.2f%%\n", k, test->num_instances(), train->num_instances(), (long long unsigned int) time_difference, num_threads, accuracy);

    free(predictions);
    free(confusionMatrix);
}
