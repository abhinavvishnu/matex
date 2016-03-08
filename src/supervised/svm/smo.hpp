#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#define EPS 1e-20
#define ZERO 1e-12

#include <mpi.h>

#include <iostream>
#include <string>
#include <iomanip>
#include <locale>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>

#include <cassert>
#include <fstream>
#include <climits>

#include "dataset.hpp"
#include "distance.hpp"

using namespace std;

class SMO {
    private:
        Comm *comm;
        Distance distance;
        Dataset *trainset, *testset;
    public:
        int offset; // the number of extra variables: 3 in case of SMO
        double b_up, b_low;
        size_t i_up, i_low;
        size_t nsv, bsv, zsv;
        size_t user_iter;
        
        double pone, mone;
        double C, sigmasqr, TOL;
        vector<double> ts_sample1, ts_sample2;

        vector<size_t> recv_row_ptr; // buffer for row pointer: should hold the data for any process
        vector<size_t> updated_gradient;
        vector<double> recv_samples; // buffer for samples: should hold the data for any process
        vector<double> temp_gradient;





        double addFcache();
        double normw();
        double addarrele();
        double evali(); 
        void fcache_reconst();
        void run();
        int takestep(size_t, size_t);
        void get_ts_samples(size_t, size_t);

        void fill_sample(size_t index, vector<double> &buffer);
        void  updategradient(size_t i1, size_t i2, double a1, double a2, double delalph1,
                double delalph2, double y1, double y2);

        SMO();
        SMO(int*, char***);

        void testing_init();
        int exchange_samples_and_row_ptr(int);
        void testing(double);

};

