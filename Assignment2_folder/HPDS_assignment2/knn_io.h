#ifndef KNN_IO_H
#define KNN_IO_H

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "libarff/arff_data.h"

inline int positive_integer(const char* text) {
    char* end = NULL;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno || end == text || *end || value < 1 || value > INT_MAX)
        throw std::runtime_error("Expected a positive integer");
    return static_cast<int>(value);
}

// Validate outside the timed region, including the parser's numeric class encoding.
inline void validate_datasets(ArffData* train, ArffData* test, int k) {
    const int attributes = train->num_attributes();
    if (attributes < 2 || attributes != test->num_attributes() ||
        train->num_instances() < k || k > INT_MAX / 2 || test->num_instances() < 1)
        throw std::runtime_error("Incompatible datasets or k exceeds training count");
    // The supplied parser uses int products when flattening the matrix.
    if (train->num_instances() > INT_MAX / attributes ||
        test->num_instances() > INT_MAX / attributes)
        throw std::runtime_error("Dataset exceeds parser indexing capacity");
    for (ArffData* data : {train, test}) {
        for (int row = 0; row < data->num_instances(); ++row) {
            for (int col = 0; col < attributes; ++col) {
                auto* field = data->get_instance(row)->get(col);
                float value = field->operator float();
                if (field->missing() || !std::isfinite(value))
                    throw std::runtime_error("Missing or non-finite dataset value");
                if (col == attributes - 1 &&
                    (value < 0 || value >= static_cast<float>(INT_MAX) ||
                     std::floor(value) != value))
                    throw std::runtime_error("Classes must be nonnegative integer labels");
            }
        }
    }
}

inline void write_predictions(const char* path, const int* predictions, int count) {
    FILE* file = std::fopen(path, "w");
    if (!file) throw std::runtime_error("Cannot open prediction output");
    bool failed = false;
    for (int i = 0; i < count; ++i)
        if (std::fprintf(file, "%d\n", predictions[i]) < 0) { failed = true; break; }
    if (std::fclose(file) != 0) failed = true;
    if (failed) throw std::runtime_error("Cannot write prediction output");
}

#endif
