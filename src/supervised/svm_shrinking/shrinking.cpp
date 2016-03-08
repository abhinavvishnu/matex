#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <string>
#include <limits>
#include <climits>
#include <map>
#include <algorithm>
#include "smo.hpp"


void SMO::shrinking_init() {
    
    shrinkiter = 0.05 * trainset->numglobalsamples();
    shrinkitercounter = 0;

    // keep track of working set
    working_set.resize(trainset->row_ptr.size());

    // initialize each sample to be active
    std::fill(working_set.begin(), working_set.end(), 1);

    // change the number of elements in working set
    n_working_set = trainset->row_ptr.size();

    // find the maximum number of samples
    size_t max_samples_size = trainset->samples.size();
    size_t max_row_ptr_size = trainset->row_ptr.size();    

    // find the maximum number of samples
    MPI_Allreduce(MPI_IN_PLACE, &max_samples_size, 1, MPI_LONG, MPI_MAX, comm->worldcomm());
    
    // find the maximum row ptr size
    MPI_Allreduce(MPI_IN_PLACE, &max_row_ptr_size, 1, MPI_LONG, MPI_MAX, comm->worldcomm());
    
    // allocate recv samples buffer
    recv_samples.resize(max_samples_size);
    
    // allocate recv row ptr buffer
    recv_row_ptr.resize(max_row_ptr_size); 
    
    // allocate a temp gradient buffer for storing temporary values
    temp_gradient.resize(trainset->row_ptr.size());

    // initialize its values to 0
    std::fill(temp_gradient.begin(), temp_gradient.end(), 0);

    // resize updated gradient
}

void SMO::reset_shrink_ds() {
    n_working_set = trainset->row_ptr.size();
    std::fill(working_set.begin(), working_set.end(), 1);
    std::fill(temp_gradient.begin(), temp_gradient.end(), 0);
    shrinkitercounter = 0;
}


int SMO::exchange_samples_and_row_ptr(int index) {
    MPI_Request req[4]; // send/recv samples, send/recv row pointers
    int nelems_in_row_ptr = 0;
    
    int send_partner = (comm->rank() + index) % comm->size(); // This can be changed, as needed
    int recv_partner = (comm->rank() + comm->size() - index) % comm->size(); // This can be changed, as needed

    void *send_samples_ptr, *recv_samples_ptr;
    void *send_rptr, *recv_rptr;

    send_samples_ptr = static_cast<void *>(&(trainset->samples[0]));
    recv_samples_ptr = static_cast<void *>(&(recv_samples[0]));

    send_rptr = static_cast<void *>(&(trainset->row_ptr[0]));
    recv_rptr = static_cast<void *>(&(recv_row_ptr[0]));
   
    int rc;

    rc = MPI_Irecv(recv_rptr, recv_row_ptr.size(), MPI_LONG,
            recv_partner, 0, comm->worldcomm(), &(req[0]));
    assert(rc == MPI_SUCCESS);

    rc = MPI_Irecv(recv_samples_ptr, recv_samples.size(), MPI_DOUBLE,
            recv_partner, 1, comm->worldcomm(), &(req[1]));
    assert(rc == MPI_SUCCESS);
    
    rc = MPI_Isend(send_rptr, trainset->row_ptr.size(), MPI_LONG,
            send_partner, 0, comm->worldcomm(), &(req[2]));
    assert(rc == MPI_SUCCESS);
    
    rc = MPI_Isend(send_samples_ptr, trainset->samples.size(), MPI_DOUBLE,
            send_partner, 1, comm->worldcomm(), &(req[3]));
    assert(rc == MPI_SUCCESS);

    MPI_Status status[4];
    rc = MPI_Waitall(4, req, status);
    assert(rc == MPI_SUCCESS);
    
    // use mpi get count
    rc = MPI_Get_count(&(status[0]), MPI_LONG, &nelems_in_row_ptr); 
    assert(rc == MPI_SUCCESS);
   
    int nelems_in_samples;
    rc = MPI_Get_count(&(status[1]), MPI_DOUBLE, &nelems_in_samples); 
    assert(rc == MPI_SUCCESS);

    assert(nelems_in_samples == recv_row_ptr[nelems_in_row_ptr - 1]);

    // debug statements
#if 0
    for (int i = 0; i < comm->size(); ++i) {
        if (i == comm->rank()) {
            cout << "rank: " << comm->rank() << endl;

            size_t rptr_count = 0;
            cout << "row_ptr:" << endl;
            for (size_t j = 0; j < nelems_in_row_ptr; j++) {
                cout << recv_row_ptr[j] << " ";
            }
            cout << endl;
            cout << "samples:" << endl;
            for (size_t j = 0; j < nelems_in_samples; j++) {
                cout << recv_samples[j] << " ";

                if (j == recv_row_ptr[rptr_count] - 1) {
                    cout << endl;
                    ++rptr_count;
                }
            }
            cout << endl;
        }
        MPI_Barrier(comm->worldcomm());
    }
    
#endif
    return nelems_in_row_ptr;
} 

void SMO::write_gradient_and_update_ds() {
    size_t nelems_in_local_sample, local_sample_offset;
#if 0
    for (int i = 0; i < comm->size(); ++i) {
        if (i == comm->rank()) {
            cout << "rank: "<< comm->rank() << ", n_shrunk: " << trainset->row_ptr.size() - n_working_set << endl;
            for (size_t k = 0; k < trainset->row_ptr.size(); k++) {
                if (temp_gradient[k])
                    cout << "rank: " << comm->rank() << ", index: " << k <<", gradient: " << temp_gradient[k] << endl; 
            }

        }
        MPI_Barrier(comm->worldcomm());
    }
#endif


    for (size_t k = 0; k < trainset->row_ptr.size(); k++) {
        if (k == 0) {
            nelems_in_local_sample = trainset->row_ptr[k];
            local_sample_offset = 0;
        }
        else {
            nelems_in_local_sample = trainset->row_ptr[k] - trainset->row_ptr[k - 1];
            local_sample_offset = trainset->row_ptr[k - 1];
        }

        double local_sample_alpha = trainset->samples[local_sample_offset + nelems_in_local_sample - 4];

        //if (!working_set[k] && local_sample_alpha < ZERO && fabs(local_sample_alpha - C) < ZERO) {
        if (!working_set[k] && (local_sample_alpha < ZERO || fabs(local_sample_alpha - C) < ZERO)) {

        trainset->samples[local_sample_offset + nelems_in_local_sample - 2]  =
                temp_gradient[k] - 
                trainset->samples[local_sample_offset + nelems_in_local_sample - 1];
        }

        double sinfo = trainset->samples[local_sample_offset + nelems_in_local_sample - 3];
        double gradient = trainset->samples[local_sample_offset + nelems_in_local_sample - 2]; 
        if (sinfo == 0 || sinfo == 3|| sinfo ==4) {
            if (gradient > b_low) {
                b_low = gradient;
                i_low = trainset->ds_start_index + k;
            }
        }

        if (sinfo == 0 || sinfo == 1 || sinfo ==2) {
            if (gradient < b_up) {
                b_up = gradient;
                i_up = trainset->ds_start_index + k;
            }

        }
    }
    // use the function now
    find_global_bup_blow();
}


void SMO::reconstruct_gradient_serial() {

}
void SMO::reconstruct_gradient() {
    // for each process do : communicate with your neighbors in increasing
    // order

    size_t nelems_in_recvd_sample, recvd_sample_offset;
    size_t nelems_in_local_sample, local_sample_offset;

    std::fill(temp_gradient.begin(), temp_gradient.end(), 0);

    double t_comm = 0, t_start, t_end;
    double gradient_start = comm->timestamp();
    for (int i = 0; i < comm->size(); i++) {
        t_start = comm->timestamp();
        int nelems_in_row_ptr = exchange_samples_and_row_ptr(i); 
        t_comm += comm->timestamp() - t_start;
        
        for (int j = 0; j < nelems_in_row_ptr; ++j) {
            if (j == 0) {
                nelems_in_recvd_sample = recv_row_ptr[j];
                recvd_sample_offset = 0;
            }
            else {
                nelems_in_recvd_sample = recv_row_ptr[j]  - recv_row_ptr[j - 1];
                recvd_sample_offset = recv_row_ptr[j - 1];
            } 

            double recvd_sample_alpha = recv_samples[recvd_sample_offset + nelems_in_recvd_sample - 4];
            if (recvd_sample_alpha <= ZERO) 
                continue;
                
            double recvd_sample_y =  recv_samples[recvd_sample_offset + nelems_in_recvd_sample - 1];
                recvd_sample_alpha *= recvd_sample_y; 

            // if the recvd sample has non-zero alpha, update our samples
            for (size_t k = 0; k < trainset->row_ptr.size(); ++k) {
                if (working_set[k])
                    continue;

                if (k == 0) {
                    nelems_in_local_sample = trainset->row_ptr[k];
                    local_sample_offset = 0;
                }
                else {
                    nelems_in_local_sample = trainset->row_ptr[k] - trainset->row_ptr[k - 1];
                    local_sample_offset = trainset->row_ptr[k - 1];
                }

                double local_sample_alpha =
                    trainset->samples[local_sample_offset +
                    nelems_in_local_sample - 4];

                assert(!(local_sample_alpha > ZERO && fabs(local_sample_alpha - C) > ZERO));

                // keep the temporary gradient
                double *recv_sample_ptr = static_cast<double *>(&(recv_samples[recvd_sample_offset]));
                double *local_sample_ptr = static_cast<double *>(&(trainset->samples[local_sample_offset]));

                // FIXME
                temp_gradient[k] += (recvd_sample_alpha *
                    distance.rbf(recv_sample_ptr,
                            nelems_in_recvd_sample - 4,
                            local_sample_ptr,
                            nelems_in_local_sample - 4, sigmasqr));

            }

        }

    }

    double gradient_time = comm->timestamp() - gradient_start;

    MPI_Barrier(comm->worldcomm());
    write_gradient_and_update_ds();
    
    if (0 == comm->rank()) {
        cout << " Completed Gradient Reconstruction" << endl;
        cout << " Actual gradient time: "<< (gradient_time - t_comm) * 1.0e3 << " Communication time: " << t_comm * 1.0e3 << endl;
    }
}

bool SMO::shrink_sample(double sinfo, double old_bup, double old_blow, double gradient, size_t index) {
    if (shrinkitercounter <  shrinkiter)
        return false;

    bool flag = false;
    if (sinfo == 3 || sinfo == 4) {
        if (gradient < old_bup)
            flag = true;
    }
    
    if (sinfo == 1 || sinfo == 2) {
        if (gradient > old_blow)
            flag = true;
    }

    return flag;
}


void SMO::update_shrinking_stats() {

    // return in this case
    if (shrinkitercounter != shrinkiter)
        return;

    size_t max_working_set, min_working_set, total_working_set;
    int rc;

    // find the max min and average number of samples owned by each process
    rc = MPI_Allreduce(&n_working_set, &max_working_set, 1, MPI_LONG, MPI_MAX, comm->worldcomm());
    rc = MPI_Allreduce(&n_working_set, &min_working_set, 1, MPI_LONG, MPI_MIN, comm->worldcomm());
    rc = MPI_Allreduce(&n_working_set, &total_working_set, 1, MPI_LONG, MPI_SUM, comm->worldcomm());

    // update the min shrink counter to the minimum of the total working set
    if (shrinkitercounter > total_working_set)
        shrinkitercounter = total_working_set;


    static double time_spent = 0;
    static bool first_time = true;

    if (first_time == true) {
        first_time = false;
        time_spent = comm->timestamp();
    } else
        time_spent = comm->timestamp() - time_spent;
#if 0
    if (0 == comm->rank()) {
        cout << "max: " << max_working_set << ", min: " << min_working_set << ", avg: "<< 
            total_working_set/comm->size() << ", time_spent: " << time_spent * 1.0e3 << endl;
    }
    
#endif
    time_spent = comm->timestamp();

            
}



























