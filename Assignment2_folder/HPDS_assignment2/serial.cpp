#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <cmath>
#include <cstring>
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

// Implements a sequential kNN where for each candidate query an in-place priority queue is maintained to identify the nearest neighbors
int* KNN(ArffData* train, ArffData* test, int k) {    

    int num_classes = train->num_classes();
    int num_attributes = train->num_attributes();
    int train_num_instances = train->num_instances();
    int test_num_instances = test->num_instances();

    // predictions is the array where you have to return the class predicted (integer) for the test dataset instances
    int* predictions = (int*)malloc(test_num_instances * sizeof(int));

    // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
    float* candidates = (float*) calloc(k*2, sizeof(float));
    for(int i = 0; i < 2*k; i++){ candidates[i] = FLT_MAX; }

    // Pointers representing the dataset as a 2D matrix size num_instances x num_attributes
    float* train_matrix = train->get_dataset_matrix();
    float* test_matrix = test->get_dataset_matrix();

    // Stores bincounts of each class over the final set of candidate NN. Calloc initializes values with 0s
    int* classCounts = (int*)calloc(num_classes, sizeof(int));

    for(int queryIndex = 0; queryIndex < test_num_instances; queryIndex++) {
        for(int keyIndex = 0; keyIndex < train_num_instances; keyIndex++) {
            
            float dist = distance(&test_matrix[queryIndex*num_attributes], &train_matrix[keyIndex*num_attributes], num_attributes);

            // Add to our candidates
            for(int c = 0; c < k; c++){
                if(dist < candidates[2*c]) {
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for(int x = k-2; x >= c; x--) {
                        candidates[2*x+2] = candidates[2*x];
                        candidates[2*x+3] = candidates[2*x+1];
                    }
                    
                    // Set key vector as potential k NN
                    candidates[2*c] = dist;
                    candidates[2*c+1] = train_matrix[keyIndex*num_attributes + num_attributes - 1]; // class value

                    break;
                }
            }
        }

        // Bincount the candidate labels and pick the most common
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

        // Make prediction with 
        predictions[queryIndex] = max_class;

        // Reset arrays
        for(int i = 0; i < 2*k; i++){ candidates[i] = FLT_MAX; }
        memset(classCounts, 0, num_classes * sizeof(int));
    }

    free(train_matrix);
    free(test_matrix);
    free(candidates);
    free(classCounts);
    
    return predictions;
}

int main(int argc, char *argv[]) try
{
    if(argc != 4 && argc != 5)
    {
        fprintf(stderr, "Usage: ./serial train.arff test.arff k [predictions.txt]\n");
        return 1;
    }

    // k value for the k-nearest neighbors
    int k = positive_integer(argv[3]);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();
    
    validate_datasets(train, test, k);
    int* predictions = NULL;
    
    // Initialize time measurement
    auto start = std::chrono::steady_clock::now();
    
    predictions = KNN(train, test, k);
    
    // Stop time measurement
    auto end = std::chrono::steady_clock::now();

    int correct = 0;
    for (int row = 0; row < test->num_instances(); ++row)
        if (predictions[row] == test->get_instance(row)->get(test->num_attributes() - 1)->operator int32())
            ++correct;
    float accuracy = 100.0f * correct / test->num_instances();
    if (argc == 5) write_predictions(argv[4], predictions, test->num_instances());

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("The %d-NN classifier required %.6f ms. Accuracy was %.2f%%\n", k, time_ms, accuracy);
    printf("RESULT strategy=serial end_to_end_ms=%.6f accuracy=%.8f\n", time_ms, accuracy);

    free(predictions);
}
 catch (const std::exception& error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
}
