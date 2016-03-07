#include <iostream>
#include "distance.hpp"
#include <cmath>

// inner product
double Distance:: inner_product(double *first, double *second, size_t len) {
    double prod = 0;

    for (size_t i = 0; i < len; i++) {
        prod += first[i] * second[i];
    }

    return prod;
}

// calculate euclidean distance
double Distance:: euclidean(double *first, double *second, size_t len) {
    if (!len)
        return 0;

    double firstsquare = inner_product(first, first, len);
    double secondsquare = inner_product(second, second, len);
    double firstdotsecond = inner_product(first, second, len);

    double absvalue = std::abs(firstsquare + secondsquare - 2 * firstdotsecond);
    return sqrt(absvalue);
}

// add the values of src in dst 
void Distance:: addvectors(double *src, double *dst, size_t len) {
    if(!len)
        return;

    for (size_t i = 0; i < len; i++) {
        dst[i] += src[i];
    }
}

void Distance:: divvectors(double *src, size_t *divisor, size_t num_elements, size_t offset) {
    if (!num_elements)
        return;

    for (size_t i = 0; i < num_elements; i++) {
        for (size_t j = 0; j < offset; j++) {
            src[i * offset + j] /= (double) divisor[i]; 
        }
    }
}
