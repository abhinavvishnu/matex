#include <iostream>
#include <mpi.h>
#include "dataset.hpp"
#include "distance.hpp"
#include <vector>
#include <queue>
#include <map>
#include "glob.h"
#include "gen.h"
#include <string.h>
#include <stdio.h>
#include <fstream>

typedef long int64_t;


size_t fpmerge(vector<double> &t1, size_t size1, vector<size_t> &freq1, vector<double> & t2, size_t size2,
            vector<size_t> &freq2, vector<double> &result, vector<size_t> & resultFreq, map<double, size_t> &item_map);

size_t prune_local_prefix_tree(vector<double> & orig_tree, size_t tree_size, vector<size_t> &freq1,
                vector<double> & pruned_tree, vector<size_t> & pruned_freq, vector<double> & label_list, size_t
                        list_size);

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
        static const size_t extra_elems = 0; // for alpha, setinfo and Fcache
        
        
        static const double ROOT = -2;
        static const double delim = -1;
        static const size_t max_sample_size = 2560;
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

        size_t max_item_id;
        // keeps frequency of all items
        vector<size_t> item_freq_count;

        // keeps list of frequent items (unsorted)
        vector<freq_item_list_t> freq_itemid_list; // sorted

        vector<double> pruned_freq_itemid_list;
        // maps the frequent items to their ranks 

        // neighbor list for phase 1 and phase 2 (optional)
        int phase1_leftn, phase1_rightn, phase2_leftn, phase2_rightn;
        size_t phase1_steps, phase2_steps;

        vector<size_t> local_freq_tree_a;
        vector<size_t> local_freq_tree_b;
        vector<double> local_item_tree_a;
        vector<double> local_item_tree_b;
        size_t local_prefix_tree_size;
        size_t global_prefix_tree_phase1_size;

        // work size plicy
        size_t matex_arm_work_size_policy;
        size_t matex_arm_work_stealing_policy;
        size_t matex_arm_work_size;

        // termination policy
        size_t matex_arm_termination_policy;
        size_t matex_arm_work_distribution_policy;
    
            void sort_sample(size_t, size_t, double *, size_t &);
    public:
        map<double, size_t> freq_item_rank;
        FPG();
        ~FPG();
        FPG(int *, char ***);
        void usage();
        void run();
        void find_frequent_ones();
        void build_local_prefix_tree();
        void build_local_prefix_tree_lb();
        void build_global_prefix_tree_phase1();
        void build_global_prefix_tree_phase2();
        void initialize_buffers(vector<double> &sample_copy, vector<size_t> &freq_copy,
                        size_t &num_global_freq_items );
        void create_mpi_windows(size_t *, double *, size_t *, MPI_Win &, MPI_Win &,MPI_Win &);
        bool select_victim_and_fill_buffers(vector<pair<size_t, size_t> > &, int &,size_t &, size_t, 
                size_t *, double *, size_t *, 
                MPI_Win &, MPI_Win &,MPI_Win &);
        bool select_victim_and_fill_buffers(vector<pair<size_t, size_t> > &, int &,size_t &, size_t, 
                size_t *, double *, size_t *, 
                void **, void **, void **);
        void initialize_mpi_buffers(size_t **, double **, size_t **);
        void initialize_comex_buffers(size_t **, double **, size_t **, void**, void**, void**);
        void sort_sample_lb(double *, size_t , double *, size_t &);
        size_t merge_samples_to_local_tree(size_t victim, size_t row_ptr_offset,
                        size_t *get_row_ptr, double *get_data_ptr, size_t *get_ws_ptr, vector<double> &sample_copy, vector<size_t> &freq_copy); 

        size_t worksize(size_t, size_t);
        void read_params();
        void setup_initial_conditions(size_t , size_t &, vector<pair<size_t, size_t> > &);

};
