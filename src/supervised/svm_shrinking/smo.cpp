#include <mpi.h>
#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <string>
#include <limits>
#include <climits>
#include <map>
#include <algorithm>
#include "smo.hpp"

using namespace std;

SMO::SMO() {
    comm = NULL;
    trainset = NULL;
    testset = NULL;
    C = 1.0;
    sigmasqr = 1.0;
    TOL = 1.0e-3;
}

SMO::SMO(int *argc, char ***argv) {
    char **tmp = *argv;
    
    comm = new Comm(argc, argv);

    if(*argc != 5) {
        if (comm->rank() == 0)
            cout << "Usage: mpirun -np $nproc ./smo trainset testset C sigmasqr" << endl;
        MPI_Barrier(comm->worldcomm());
        MPI_Abort(comm->worldcomm(), -1);
    }
    
    trainset = new Dataset(tmp[1], comm->rank());
    testset = new Dataset(tmp[2], comm->rank());

    C = atof(tmp[3]);
    sigmasqr = atof(tmp[4]);
    TOL = 1.0e-3;
    pone = 1, mone = -1, offset = 3;

    int root = 0;
    size_t train_samples, test_samples;

    if (0 == comm->rank()) {
        train_samples = trainset->numglobalsamples();
        test_samples = testset->numglobalsamples();
    }

    // broadcast the number of training samples
    MPI_Bcast(&train_samples, sizeof(size_t), MPI_BYTE, root,  comm->worldcomm());
    
    // broadcast the number of testing samples
    MPI_Bcast(&test_samples, sizeof(size_t), MPI_BYTE, root,  comm->worldcomm());

    // assert that number of global samples is geq nproc
    assert(train_samples >= comm->size() && test_samples >= comm->size());

    // set the correct values after broadcast
    trainset->setnumglobalsamples(train_samples);
    testset->setnumglobalsamples(test_samples);

    // create extra space for alpha, setinfo and gradient
    size_t extra_elems = 3;
    if(trainset->readsparse(comm, extra_elems) == false) {
        cout << "Error in reading the training set file" << endl;
    }

    // for testing set, this information is not needed
    if(testset->readsparse(comm, 0) == false) {
        cout << "Error in reading the testing set file" << endl;
    }
   
    // Broadvast the max sample size
    MPI_Bcast(&(trainset->max_sample_size), sizeof(size_t), MPI_BYTE, root, comm->worldcomm());
    
    // resize the takestep samples, extra element is for the size
    ts_sample1.resize(trainset->max_sample_size + 1);  
    ts_sample2.resize(trainset->max_sample_size + 1);
   
    // set the local and global indices for my rank 
    trainset->setindices(comm);
    
    // munge the dataset such that alpha, s, f and y are adjoining
    trainset->munge(comm);

    trainset->row_ptr_size = trainset->row_ptr.size();
    trainset->samples_size = trainset->samples.size();
    testset->row_ptr_size = testset->row_ptr.size();
    testset->samples_size = testset->samples.size();

    // setup default values of b_up and b_low
    b_up = mone, b_low = pone;
   
    // setup default values of i_up and i_low 
    i_low = trainset->i_low;
    i_up = trainset->i_up;

    if (comm->rank() == 0) {
        cout << "MaTEx Support Vector Machine Algorithm (with shrinking):" << endl;
    }

    // initialize threshold(s) and buffers for shrinking
    shrinking_init();

    MPI_Barrier(comm->worldcomm());
}

void SMO::fill_sample(size_t index, vector<double> &buffer) {
    size_t local_index = index - trainset->ds_start_index;
    
    size_t offset, nelems;
    if (local_index == 0) {
        offset = 0;
        nelems = trainset->row_ptr[local_index];
    }
    else {
        offset = trainset->row_ptr[local_index - 1];
        nelems = trainset->row_ptr[local_index] - trainset->row_ptr[local_index - 1];
    }
    
    buffer[0] = nelems + 1;
    for (size_t j = 0; j < nelems; j++) {
        buffer[j + 1] = trainset->samples[offset + j];
    }


}

void SMO::get_ts_samples(size_t i1, size_t i2) {
    int root1, root2;

    root1 = trainset->get_rank_from_index(comm, i1);
    root2 = trainset->get_rank_from_index(comm, i2);

    if (root1 == comm->rank()) {
        fill_sample(i1, ts_sample1);
    }
    MPI_Bcast(static_cast<void *>(&(ts_sample1[0])), trainset->max_sample_size + 1, MPI_DOUBLE, root1, comm->worldcomm());
    
    if (root2 == comm->rank()) {
        fill_sample(i2, ts_sample2);
    }
    
    MPI_Bcast(static_cast<void *>(&(ts_sample2[0])), trainset->max_sample_size + 1, MPI_DOUBLE, root2, comm->worldcomm());

#if 0
    for(size_t i = 0; i < comm->size(); i++) {
        if (i == comm->rank()) {
            cout << "ts_sample1 size:" << ts_sample1[0] << ", ts_sample2 size" << ts_sample2[0] << endl;
            for (size_t j = 0; j < ts_sample1[0]; j++) {
                cout << ts_sample1[j]<< " ";
                }
                cout << endl;

            for (size_t j = 0; j < ts_sample2[0]; j++) {
                cout << ts_sample2[j]<< " ";
            }
            cout << endl;
        }
        MPI_Barrier(comm->worldcomm());
    }
#endif

}

void SMO::find_global_bup_blow() {
#if 0
    double out_low, out_up;
    MPI_Allreduce(&b_low, &out_low, 1, MPI_DOUBLE, MPI_MAX, comm->worldcomm());
    MPI_Allreduce(&b_up, &out_up, 1, MPI_DOUBLE, MPI_MIN, comm->worldcomm());

    if (fabs(out_low - b_low) > 1e-10) {
        i_low = LONG_MAX;
    }

    if (fabs(out_up - b_up) > 1e-10) {
        i_up = LONG_MAX;
    }

    MPI_Allreduce(MPI_IN_PLACE, &i_low, 1, MPI_LONG, MPI_MIN, comm->worldcomm());
    MPI_Allreduce(MPI_IN_PLACE, &i_up, 1, MPI_LONG, MPI_MIN, comm->worldcomm());
    b_low = out_low;
    b_up = out_up;
#else
    double in[4], out[4];
    in[0] = b_up;
    in[1] = b_low;
    in[2] = i_up;
    in[3] = i_low;
    int rc = MPI_Allreduce(in, out, 4, MPI_DOUBLE, comm->mpi_op, comm->worldcomm());
    assert(rc == MPI_SUCCESS);
    double out_up= out[0];
    double out_low = out[1];
    if (fabs(out_low - b_low) > 1e-10) {
        i_low = LONG_MAX;
    }

    if (fabs(out_up - b_up) > 1e-10) {
        i_up = LONG_MAX;
    }
    MPI_Allreduce(MPI_IN_PLACE, &i_low, 1, MPI_LONG, MPI_MIN, comm->worldcomm());
    MPI_Allreduce(MPI_IN_PLACE, &i_up, 1, MPI_LONG, MPI_MIN, comm->worldcomm());
    b_low = out_low;
    b_up = out_up;
#endif

}

void SMO::updategradient(size_t i1, size_t i2, double a1, double a2, double delalph1,
                        double delalph2, double y1, double y2) {

    double upmin = 1.0e12, lowmax = -1.0e12;    
    double *ts1_ptr, *ts2_ptr;

    double old_bup = b_up, old_blow = b_low;
    b_up = upmin, b_low = lowmax;
   
    size_t nelems_sample, offset;
    ts1_ptr = static_cast<double *>(&(ts_sample1[1]));
    ts2_ptr = static_cast<double *>(&(ts_sample2[1]));

    // increment the shrink itercounter
    shrinkitercounter = (shrinkitercounter + 1) % (shrinkiter + 1);

    double t_start = comm->timestamp();
    for (size_t i = 0; i < trainset->row_ptr.size(); i++) {
        // skip this sample, if inactive
        if (!working_set[i])
            continue;

        if (i == 0) {
            nelems_sample = trainset->row_ptr[0];
            offset = 0;
        }
        else {
            nelems_sample = trainset->row_ptr[i] - trainset->row_ptr[i - 1];
            offset = trainset->row_ptr[i - 1];
        }
        size_t global_index = trainset->ds_start_index + i; 
       // update alpha1
       if (global_index == i1) {
           trainset->samples[offset + nelems_sample - 4] = a1;
       }
       // update alpha2
       if (global_index == i2) {
           trainset->samples[offset + nelems_sample - 4] = a2;
        }
       
       // update setinfo
       if (global_index == i1 || global_index == i2) {  
           double alphval, yval;
           alphval = trainset->samples[offset + nelems_sample - 4];
           yval = trainset->samples[offset + nelems_sample - 1];
           if (alphval > ZERO && fabs(alphval - C ) > ZERO)
               trainset->samples[offset + nelems_sample - 3] = 0;

           if (alphval < ZERO ) {
               if ( fabs(yval - pone) < ZERO )
                   trainset->samples[offset + nelems_sample - 3] = 1;
               else
                   trainset->samples[offset + nelems_sample - 3] = 4;
           }

           if ( fabs(alphval - C) < ZERO) {
               if ( fabs(yval - pone) < ZERO)
                   trainset->samples[offset + nelems_sample - 3] = 3;
               else
                   trainset->samples[offset + nelems_sample - 3] = 2;
           }
       } 
       // compute kernel
       double withone, withtwo;
       double *sample_ptr = static_cast<double *>(&(trainset->samples[offset]));
       withone = distance.rbf(sample_ptr, nelems_sample - 4, ts1_ptr, ts_sample1[0] - 5, sigmasqr); 
       withtwo = distance.rbf(sample_ptr, nelems_sample - 4, ts2_ptr, ts_sample2[0] - 5, sigmasqr); 

       // Update gradient
       trainset->samples[offset + nelems_sample - 2] += y1 * delalph1 * withone +
           y2 * delalph2 * withtwo;

       double gradient = trainset->samples[offset + nelems_sample - 2];

       double sinfo = trainset->samples[offset + nelems_sample - 3];
       if (sinfo == 0 || sinfo == 1 || sinfo == 2 )
       {
           if (gradient < upmin)
           {
               upmin = gradient;
               b_up = upmin;
               i_up = global_index;
           }
       }

       if (sinfo == 0 || sinfo == 3 || sinfo == 4 )
       {
           if (gradient > lowmax)
           {
               lowmax = gradient;
               b_low = lowmax;
               i_low = global_index;
           }
       }
       // shrink this sample, if necessary
       bool shrunk_sample = shrink_sample(sinfo, old_bup, old_blow, gradient, i);
    
        if (working_set[i] && shrunk_sample) {
            working_set[i] = 0; // eliminate the sample for next iteration 
            n_working_set--;
        }
    }

    // update shrinking statistics
    update_shrinking_stats();
    double t_gradient = comm->timestamp();

    // find global bup and blow
    find_global_bup_blow();
    double t_end = comm->timestamp();

    static size_t print_counter = 0;

    if (0 == comm->rank() && print_counter++ % 500 == 0) 
        cout << "b_up: " << b_up << ", b_low: " << b_low<<  endl;
}

int SMO::takestep(size_t i1, size_t i2) {
    if (i1 == i2) {
        return 0;
    }
        
    double s, alph1, alph2, F1, F2, L, H, k11, k22, k12, y1, y2 ;
    double eta, a2, a1, delalph1, delalph2, compart, Lobj, Hobj;
    double t;

    double t_start = comm->timestamp();

    // fill in the buffers for ts_sample1 and ts_sample2
    get_ts_samples(i1, i2);

    double t_bcast = comm->timestamp();

    // fill stack variables
    size_t nelems_ts1, nelems_ts2;
    nelems_ts1 = ts_sample1[0], nelems_ts2 = ts_sample2[0];
    
    y1 = ts_sample1[nelems_ts1 - 1];
    F1 = ts_sample1[nelems_ts1 - 2];
    alph1 = ts_sample1[nelems_ts1 - 4];

    y2 = ts_sample2[nelems_ts2 - 1];
    F2 = ts_sample2[nelems_ts2 - 2];
    alph2 = ts_sample2[nelems_ts2 - 4];

    // Do the math mumbo jumbo
    s = y1 * y2;
    if ( fabs(y1 - y2) > ZERO)
    {
        L = ( (alph2 - alph1) > ZERO )? alph2 - alph1 : 0.0;
        H = ( (C + alph2 - alph1) < C) ?
            C + alph2 - alph1 : C ;
    }
    else
    {
        L = ( (alph2 + alph1 - C) > ZERO )?
            alph2 + alph1 - C : 0.0;
        H = ( (alph2 + alph1) < C) ? alph2 + alph1 : C ;
    }
    if ( fabs(L - H) < ZERO )
    {
        return 0;
    }
    double *ts1_ptr = static_cast<double *>(&(ts_sample1[1]));
    double *ts2_ptr = static_cast<double *>(&(ts_sample2[1]));
    
    //assert(nelems_ts1 - 5 && nelems_ts2 - 5);
    assert(ts1_ptr && ts2_ptr);
    
    k11 = distance.rbf(ts1_ptr, nelems_ts1 - 5, ts1_ptr, nelems_ts1 - 5, sigmasqr);
    k22 = distance.rbf(ts2_ptr, nelems_ts2 - 5, ts2_ptr, nelems_ts2 - 5, sigmasqr);
    k12 = distance.rbf(ts1_ptr, nelems_ts1 - 5, ts2_ptr, nelems_ts2 - 5, sigmasqr);
    
    eta = 2 * k12 - k11 - k22;
    if ( eta < ZERO)
    {
        a2 = alph2 - y2 * (F1 - F2) / eta;
        if (a2 < L)
        {
            a2 = L;
        }
        else if (a2 > H)
            a2 = H;
    }
    else
    {
        compart = y2 * (F1 - F2) - eta * alph2;
        Lobj = 0.5 * eta * L * L + compart * L;
        Hobj = 0.5 * eta * H * H + compart * H;
        if (Lobj > Hobj + EPS)
            a2 = L;
        else if (Lobj < Hobj - EPS)
            a2 = H;
        else
            a2 = alph2;
    }
    if ( a2 < ZERO)
    {
        a2 = 0.0;
    }
    else if (a2 > (C - ZERO))
        a2 = C;

    delalph2 = a2 - alph2;
    if ( fabs(delalph2) < EPS * (a2 + alph2 + EPS ))
    {
        cout <<"fabs quit" << endl;
        assert(0);
    }
    a1 = alph1 + s * (alph2 - a2);
    if ( a1 < ZERO)
    {
        a2 += s * a1;
        a1 = 0.0;
    }
    else if( a1 >(C - ZERO) )
    {
        t = a1 - C;
        a2 += s * t;
        a1 = C;
    }

    delalph1 = a1 - alph1;

    // Updgrade gradients
   
    double t_takestep = comm->timestamp();
    updategradient(i1, i2, a1, a2, delalph1, delalph2, y1, y2);
#if 0
    static size_t print_counter = 0;
    if (0== comm->rank() && print_counter++ % 500 == 0)
        cout << "Time bcast:" << (t_bcast - t_start) * 1.0e3 << ", t_takestep: " << (t_takestep - t_bcast) * 1.0e3 << endl;
#endif
    return 1;
}

double SMO::addFcache() {
    // nsv: number of support vectors
    double fcache_sum = 0;
    size_t count = 0;    

    // initialize
    nsv = 0, bsv = 0, zsv = 0;
    size_t nelems_sample, offset;
    for (size_t i = 0; i < trainset->row_ptr.size(); i++) {
        if (i == 0) {
            nelems_sample = trainset->row_ptr[0];
            offset = 0;
        }
        else {
            nelems_sample = trainset->row_ptr[i] - trainset->row_ptr[i - 1];
            offset = trainset->row_ptr[i - 1];
        }

        double alphaval = trainset->samples[offset + nelems_sample - 4];
        if (alphaval > ZERO && fabs(alphaval - C) > ZERO) {
            fcache_sum += trainset->samples[offset + nelems_sample - 2];
            ++count;
        }
        if (alphaval > ZERO && fabs(alphaval - C) > ZERO) 
            nsv++;
    }

    MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_LONG, MPI_SUM, comm->worldcomm());
    MPI_Allreduce(MPI_IN_PLACE, &nsv, 1, MPI_LONG, MPI_SUM, comm->worldcomm());
    MPI_Allreduce(MPI_IN_PLACE, &fcache_sum, 1, MPI_DOUBLE, MPI_SUM, comm->worldcomm());
    
    if (count) 
        return (fcache_sum/count);
    else 
        return ((b_up + b_low) * 0.5);
}

void SMO::run() {
    size_t i1, i2;

    MPI_Barrier(comm->worldcomm());
    
    double t_start = comm->timestamp(); // returns in seconds

    double gradient_stime, gradient_etime, gradient_ttime = 0;
    int outerexit = 0;
    int twentytolflag = 0;
    int takestepfail = 0;

    while(!outerexit) {
        while (b_up < b_low - 2 * TOL || !takestepfail) {
            i2 = i_low;
            i1 = i_up;
            takestepfail = takestep(i1, i2);
            if (b_low - b_up < 20 *TOL && !twentytolflag) {
                
                gradient_stime = comm->timestamp(); 
                //reconstruct_gradient();
                gradient_etime = comm->timestamp();
                gradient_ttime += gradient_etime - gradient_stime;

                twentytolflag = 1; 
                //reset_shrink_ds();
            }
        }
        takestepfail = 0;
        gradient_stime = comm->timestamp(); 
        reconstruct_gradient();
        gradient_etime = comm->timestamp();
        gradient_ttime += gradient_etime - gradient_stime;
        reset_shrink_ds();

        if (b_low - b_up > 2 * TOL)
            outerexit = 0;
        else
            outerexit = 1;
    }

    double thresh = addFcache();
    MPI_Barrier(comm->worldcomm());
    double t_end = comm->timestamp(); // returns in seconds
    if (0 == comm->rank()) {
        cout <<"Threshold: " << thresh << endl;;
        cout <<"Nsv: " << nsv << ", Bsv: " << bsv << ", Zsv: "<< zsv << endl;;
        cout << "Total Time: " << (t_end - t_start) * 1.0e3 << " ms" << endl;
//        cout << "Gradient Time: " << (gradient_ttime) * 1.0e3 << " ms" << endl;
    }    

    testing_init();
    testing(thresh);
}

