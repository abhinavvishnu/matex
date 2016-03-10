
#include "comm.hpp"
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;
class Dataset {
    private:
        char *filename;
    public:
        size_t num_local_samples;
        size_t num_global_samples;
        size_t row_ptr_size; // cache the values since the vectors will be resized later
        size_t samples_size;
        size_t max_sample_size;
        size_t ds_start_index, ds_end_index;
        size_t i_up, i_low;

        ifstream inp_file;
        vector<size_t> row_ptr;
        vector<double> samples;
        Dataset(char *, int);
        Dataset();
        ~Dataset();
        bool readsparse(Comm *, size_t);
        void print(Comm *);
        void setnumglobalsamples(size_t);
        size_t numglobalsamples() const;
        size_t numlocalsamples() const;

        void exchange_data(Comm *, size_t&, size_t&, int ) ;

        size_t calculate_global_index(Comm *, size_t);
        void setindices(Comm *comm);
        int get_rank_from_index(Comm *comm, size_t local_index);

        void munge(Comm *);
        // Helper Functions
};
