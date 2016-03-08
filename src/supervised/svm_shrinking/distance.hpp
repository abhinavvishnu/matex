#include <iostream>
#include <vector>

class Distance {

    public:
        double euclidean(double *, double *, size_t);
        double euclideansparse(double *, size_t, double *, size_t);
        double inner_product(double *, double *, size_t);
        double inner_product_sparse(double *, size_t, double *, size_t);
        void addvectors(double *, double *, size_t);
        void divvectors(double *, size_t *, size_t, size_t);
        double mydpfunc(double *, size_t, double *, size_t);
        double rbf(double *first, size_t firstlen, double *second, size_t secondlen, double sigmasqr);

};
