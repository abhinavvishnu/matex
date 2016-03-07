#include <iostream>
#include "distance.hpp"
#include <cmath>
#include <cassert>

using namespace std;

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

    return sqrt(firstsquare + secondsquare - 2 * firstdotsecond);
}


double Distance:: inner_product_sparse(double *first, size_t firstlen, double *second, size_t secondlen) {
    double prod = 0;

    double col1, col2, val1, val2;
    size_t firstiter = 0, seconditer = 0;
    while (firstiter < firstlen && seconditer< secondlen) {
        col1 = first[firstiter]; val1 = first[firstiter + 1];
        col2 = second[seconditer]; val2 = second[seconditer + 1];

        if ((size_t)col1 == (size_t)col2) {
            prod += (val1 -  val2) * (val1 - val2);    
            firstiter += 2; seconditer += 2;
        }
        else if ((size_t)col1 < (size_t)col2)  {
            prod += val1 * val1;
            firstiter += 2;
        }
        else {
            prod += val2 * val2;
            seconditer += 2;
        }
    }

    while (firstiter < firstlen) {
        col1 = first[firstiter]; val1 = first[firstiter + 1];
        prod += val1 * val1;
        firstiter += 2;
    }

    while (seconditer < secondlen) {
        col2 = second[seconditer]; val2 = second[seconditer + 1];
        prod += val2 * val2;
        seconditer += 2;
    }

    return prod;
}


double Distance:: euclideansparse(double *first, size_t firstlen, double *second, size_t secondlen) {

    //cout << firstlen << "," << secondlen << endl;
    assert(!(firstlen & 1));
    assert(!(secondlen & 1)); // both lengths should be even
    
    double firstdotsecond = inner_product_sparse(first, firstlen, second, secondlen);

    assert(firstdotsecond >= 0);
    return sqrt(abs(firstdotsecond));
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
