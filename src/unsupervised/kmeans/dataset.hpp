#include <iostream>
#include <vector>
#include <fstream>
#include "comm.hpp"

using namespace std;
class Dataset {
    private:
        char *filename;
        size_t num_global_samples;
        size_t num_local_samples;
        ifstream inp_file;
    public:
        vector<size_t> row_ptr;
        vector<double> samples;
        Dataset(char *, int);
        ~Dataset();
        bool readsparse(Comm *);
        void print(Comm *);
        const size_t getnumlocalsamples() const;
        void setnumlocalsamples(size_t);
        const size_t getnumglobalsamples() const;
        void setnumglobalsamples(size_t);
        void exchange_data(Comm *, size_t&, size_t&, int ) ;

        // Helper Functions
};
