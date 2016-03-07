#include <iostream>
#include <vector>

class Distance {

    public:
        double euclidean(double *, double *, size_t);
        double inner_product(double *, double *, size_t);
        void addvectors(double *, double *, size_t);
        void divvectors(double *, size_t *, size_t, size_t);
};
