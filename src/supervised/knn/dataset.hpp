#include <iostream>
#include <vector>
#include <fstream>
#include "comm.hpp"

using namespace std;
class Dataset {
    private:
        char *filename;
    public:
        size_t num_local_samples;
        size_t num_global_samples;
        size_t row_ptr_size; // cache the values since the vectors will be resized later
        size_t samples_size;

        ifstream inp_file;
        vector<size_t> row_ptr;
        vector<double> samples;
        Dataset(char *, int);
        ~Dataset();
        bool readsparse(Comm *, size_t);
        void print(Comm *);
        void setnumglobalsamples(size_t);
        const size_t numglobalsamples() const;
        const size_t numlocalsamples() const;

        void exchange_data(Comm *, size_t&, size_t&, int ) ;

        // Helper Functions
};
