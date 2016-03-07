#include <iostream>
#include <mpi.h>
#include "dataset.hpp"
#include "distance.hpp"
#include <vector>
#include <queue>
#include <map>

typedef long int64_t;


size_t fpmerge(vector<double> &t1, size_t size1, vector<size_t> &freq1, vector<double> & t2, size_t size2,
            vector<size_t> &freq2, vector<double> &result, vector<size_t> & resultFreq, map<double, size_t> &item_map);


struct output_elem {
    int classvar;
    double distance;
};

struct freq_item_list_t {
    double item_id;
    double support;
};
struct compare_by_support {
    bool operator () (const freq_item_list_t &a, const freq_item_list_t &b) const {
        return (a.support >  b.support);
    };
};
class FPG {
    private:
        double SUPPORT_COUNT;
        size_t extra_elems; // for alpha, setinfo and Fcache
        
        
        double ROOT;
        double delim;
        size_t max_sample_size;
        Dataset *dataset;
        Comm    *comm;
        
        // maximum local samples owned by any process
        size_t max_local_samples;

        // max size of row_ptr owned by any process
        size_t max_row_ptr_size;

        // max size of samples array
        size_t max_samples_size;

        vector<size_t> recv_row_ptr;
        vector<double> recv_samples;

        // priority queue for the output


        size_t max_item_id;
        // keeps frequency of all items
        vector<size_t> item_freq_count;

        // keeps list of frequent items (unsorted)
        vector<freq_item_list_t> freq_itemid_list; // sorted

        // maps the frequent items to their ranks 

        vector<size_t> local_freq_tree_a;
        vector<size_t> local_freq_tree_b;
        vector<double> local_item_tree_a;
        vector<double> local_item_tree_b;
        void sort_sample(size_t, size_t, double *, size_t &);
        size_t local_prefix_tree_size;

    public:
        map<double, size_t> freq_item_rank;
        FPG();
        ~FPG();
        FPG(int *, char ***);
        void usage();
        void run();
        void find_frequent_ones();
        void build_local_prefix_tree();
        void build_global_prefix_tree();

};

