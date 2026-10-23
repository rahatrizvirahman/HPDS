#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <bits/stdc++.h>
#include <chrono>
#include "knn_io.h"
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

int main(int argc, char *argv[]) try
{
    if(argc != 5 && argc != 6)
    {
        fprintf(stderr, "Usage: ./openmp train.arff test.arff k num_threads [predictions.txt]\n");
        return 1;
    }

    // k value for the k-nearest neighbors and the worker count
    int k = positive_integer(argv[3]);
    int num_threads = positive_integer(argv[4]);
    omp_set_num_threads(num_threads);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();
    validate_datasets(train, test, k);

    // Same classification boundary as serial.cpp, after parsing and including matrix preparation
    auto start = std::chrono::steady_clock::now();
    int* predictions = KNN(train, test, k);
    auto end = std::chrono::steady_clock::now();

    int correct = 0;
    for (int row = 0; row < test->num_instances(); ++row)
        if (predictions[row] == test->get_instance(row)->get(test->num_attributes() - 1)->operator int32())
            ++correct;
    float accuracy = 100.0f * correct / test->num_instances();
    if (argc == 6) write_predictions(argv[5], predictions, test->num_instances());

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("The %d-NN classifier required %.6f ms with %d threads. Accuracy was %.2f%%\n", k, time_ms, num_threads, accuracy);
    printf("RESULT strategy=openmp workers=%d end_to_end_ms=%.6f accuracy=%.8f\n", num_threads, time_ms, accuracy);

    free(predictions);
}
catch (const std::exception& error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
}
