#include <iostream>
#include <mpi.h>
#include "dataset.hpp"
#include "distance.hpp"
#include <vector>
#include <queue>

struct output_elem {
    int classvar;
    double distance;
};
struct compare_by_distance {
    bool operator () (const output_elem &a, const output_elem &b) const {
        return (a.distance < b.distance);
    };
};

class KNN {
    private:
        static const size_t extra_elems = 0; // for alpha, setinfo and Fcache
        int K; // the K in KNN
        
        Dataset *trainset, *testset;
        Comm    *comm;
        
        ifstream train_file;
        ifstream test_file;

        // maximum local samples owned by any process
        size_t max_local_samples;

        // max size of row_ptr owned by any process
        size_t max_row_ptr_size;

        // max size of samples array
        size_t max_samples_size;

        vector<size_t> recv_row_ptr;
        vector<double> recv_samples;

        // priority queue for the output

        vector<priority_queue<output_elem, vector<output_elem>, compare_by_distance> > pq;
    public:
        KNN();
        ~KNN();
        KNN(int *, char ***);
        void usage();
        void train();
        void test();
        void initiate_send_and_recv(int, size_t, size_t, size_t &, size_t &, double *, size_t *);
        void complete_send_and_recv(size_t &, size_t &);

};

