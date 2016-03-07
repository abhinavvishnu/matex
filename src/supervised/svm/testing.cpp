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


void SMO::testing_init() {
    
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
    temp_gradient.resize(testset->row_ptr.size());

    // initialize its values to 0
    std::fill(temp_gradient.begin(), temp_gradient.end(), 0);

    // resize updated gradient
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

void SMO::testing(double thresh) {
    // for each process do : communicate with your neighbors in increasing
    // order

    size_t nelems_in_recvd_sample, recvd_sample_offset;
    size_t nelems_in_local_sample, local_sample_offset;

    std::fill(temp_gradient.begin(), temp_gradient.end(), 0);

    double t_comm = 0, t_start, t_end;
    double gradient_start = comm->timestamp();

    double mclass = 0;

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
            for (size_t k = 0; k < testset->row_ptr.size(); ++k) {

                if (k == 0) {
                    nelems_in_local_sample = testset->row_ptr[k];
                    local_sample_offset = 0;
                }
                else {
                    nelems_in_local_sample = testset->row_ptr[k] - testset->row_ptr[k - 1];
                    local_sample_offset = testset->row_ptr[k - 1];
                }

                // keep the temporary gradient
                double *recv_sample_ptr = static_cast<double *>(&(recv_samples[recvd_sample_offset]));
                double *local_sample_ptr = static_cast<double *>(&(testset->samples[local_sample_offset + 1]));

                temp_gradient[k] += (recvd_sample_alpha *
                    distance.rbf(recv_sample_ptr,
                            nelems_in_recvd_sample - 4,
                            local_sample_ptr,
                            nelems_in_local_sample - 1, sigmasqr));

            }
        }
    }
    
    
    for (size_t k = 0; k < testset->row_ptr.size(); ++k) {
        if (k == 0) {
            nelems_in_local_sample = testset->row_ptr[k];
            local_sample_offset = 0;
        }
        else {
            nelems_in_local_sample = testset->row_ptr[k] - testset->row_ptr[k - 1];
            local_sample_offset = testset->row_ptr[k - 1];
        }
        double local_sample_y = testset->samples[local_sample_offset]; 
        temp_gradient[k] -= thresh;
        if(temp_gradient[k] < ZERO && local_sample_y > ZERO) 
            ++mclass;
        if (temp_gradient[k] > ZERO && local_sample_y < ZERO)
            ++mclass;
    }

    // now subtract the threshold and see the missed classes

    MPI_Allreduce(MPI_IN_PLACE, &mclass, 1, MPI_DOUBLE, MPI_SUM, comm->worldcomm());
    

    double gradient_time = comm->timestamp() - gradient_start;

    
    if (0 == comm->rank()) {
        cout << " Actual testing time: "<< (gradient_time - t_comm) * 1.0e3 << " Accuracy: " << (testset->numglobalsamples() - mclass)/ 
            (1.0* testset->numglobalsamples())* 100 << endl;
    }
}

























