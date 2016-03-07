#include <iostream>
#include <mpi.h>
#include "dataset.hpp"
#include "distance.hpp"
#include <vector>

class kmeans {
    private:
        Dataset *dataset;
        Comm    *comm;
        // the centroids are always dense: up to the max dimension
        vector<double> centroids;
        size_t maxdim;
        size_t num_centroids;
        int MAX_ITER;
    public:
        kmeans();
        ~kmeans();
        kmeans(int *, char ***);
        void seed();
        void iterative();
        size_t pick_local_centroids(int, int);
        void print_stats();
};


