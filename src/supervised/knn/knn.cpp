#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <string>
#include <limits>
#include <map>
#include "knn.hpp"

KNN::KNN() {
    comm = NULL;
    trainset = NULL;
    testset = NULL;
}

KNN::KNN(int *argc, char ***argv) {
    char **tmp = *argv;
    
    if (*argc != 4) {
        cout << "Usage :: mpirun -np proc_count TrainingFile TestFile K" << endl;
        exit(-1); 
    }
    comm = new Comm(argc, argv);

    trainset = new Dataset(tmp[1], comm->rank());

    testset = new Dataset(tmp[2], comm->rank());

    K = atoi(tmp[3]);
   
    // process 0 reads the number of samples 
    int root = 0;
    size_t train_samples, test_samples;
    
    if (0 == comm->rank()) {
        train_samples = trainset->numglobalsamples();
        test_samples = testset->numglobalsamples();
    }

    MPI_Bcast(&train_samples, sizeof(size_t), MPI_BYTE, root,  comm->worldcomm());
    MPI_Bcast(&test_samples, sizeof(size_t), MPI_BYTE, root,  comm->worldcomm());
    
    // set the correct values after broadcast
    trainset->setnumglobalsamples(train_samples);
    testset->setnumglobalsamples(test_samples);
     
    // read the training file and allow extra elems to be appended on each sample 
    if(trainset->readsparse(comm, extra_elems) == false) {
        cout << "Error in reading the training set file" << endl;
    }

    // read the testing file and allow extra elems to be appended on each sample 
    if(testset->readsparse(comm, extra_elems) == false) {
        cout << "Error in reading the training set file" << endl;
    }
    // cache the values
    trainset->row_ptr_size = trainset->row_ptr.size();
    trainset->samples_size = trainset->samples.size();
    testset->row_ptr_size = testset->row_ptr.size();
    testset->samples_size = testset->samples.size();

    // Find the max size of number of samples, row_ptr and samples array
    // size

    size_t data[3]; // one for each
    data[0] = trainset->numlocalsamples();
    data[1] = trainset->row_ptr.size();
    data[2] = trainset->samples.size();

    // perform allreduce
    MPI_Allreduce(MPI_IN_PLACE, data, 3, MPI_LONG, MPI_MAX, comm->worldcomm());

    // set the values
    max_local_samples = data[0];
    max_row_ptr_size = data[1];
    max_samples_size = data[2];
    
    // now resize the recv buffers
    recv_row_ptr.resize(max_row_ptr_size);
    recv_samples.resize(max_samples_size);

    // resize the row ptr and samples vector as well
    trainset->row_ptr.resize(max_row_ptr_size);
    trainset->samples.resize(max_samples_size);
    
    // resize the priority queue
    pq.resize(testset->num_local_samples);
    
}

void KNN::usage() {
    if (comm->rank() == 0) {
        cout << "Usage :: mpirun -np proc_count TrainingFile TestFile K" << endl;
    }
}

void KNN::initiate_send_and_recv(int i, size_t send_row_ptr_size, size_t
        send_samples_size, size_t & recv_row_ptr_size, size_t & recv_samples_size,
        double *comp_samples_buf, size_t *comp_rptr_buf) {
    int rc;

    double *send_samples_buf, *recv_samples_buf;
    size_t *send_rptr_buf, *recv_rptr_buf;
    int src, dst;

    // left neighbor
    src = (comm->rank() + comm->size() - 1) % comm->size();
    
    // right neighbor 
    dst = (comm->rank() + 1) % comm->size();

    if (i & 1) { // odd
        recv_samples_buf = static_cast<double *>(& (trainset->samples[0]));
        send_samples_buf = static_cast<double*>(& (recv_samples[0]));
        recv_rptr_buf = static_cast<size_t*>(& (trainset->row_ptr[0]));
        send_rptr_buf = static_cast<size_t*>(& (recv_row_ptr[0])); 
    }
    else { // even
        send_samples_buf = static_cast<double *>(& (trainset->samples[0]));
        recv_samples_buf = static_cast<double*>(& (recv_samples[0]));
        send_rptr_buf = static_cast<size_t*>(& (trainset->row_ptr[0]));
        recv_rptr_buf = static_cast<size_t*>(& (recv_row_ptr[0])); 
    }
   

    MPI_Request req[4]; // For sends and receives
    MPI_Status status[4];

    rc = MPI_Irecv(recv_samples_buf, max_samples_size, MPI_DOUBLE, src, 0, comm->worldcomm(), &(req[0]));
    assert(rc == MPI_SUCCESS);
    rc = MPI_Irecv(recv_rptr_buf, max_row_ptr_size, MPI_LONG, src, 1, comm->worldcomm(), &(req[1])); 
    assert(rc == MPI_SUCCESS);
    
    rc = MPI_Isend(send_samples_buf, send_samples_size, MPI_DOUBLE, dst, 0, comm->worldcomm(), &(req[2]));
    assert(rc == MPI_SUCCESS);
    rc = MPI_Isend(send_rptr_buf, send_row_ptr_size, MPI_LONG, dst, 1, comm->worldcomm(), &(req[3]));
    assert(rc == MPI_SUCCESS);

    rc = MPI_Waitall(4, req, status);
    assert(rc == MPI_SUCCESS);

    MPI_Get_count(&(status[0]), MPI_DOUBLE, (int *)&recv_samples_size);
    MPI_Get_count(&(status[1]), MPI_DOUBLE, (int *)&recv_row_ptr_size);
    // fill in the the buffer pointers
    
    //cout << "rank:" << comm->rank() << ", " <<  recv_samples_size << ",  " << recv_row_ptr_size << ", " << send_samples_size << ", " << send_row_ptr_size << endl;
}

void KNN::complete_send_and_recv(size_t & recv_row_ptr_size, size_t & recv_samples_size) {
    // No overlap of communication with computation at the moment, 
    // This function is noop

}


// The Algorithm rotates around the training set across all processes using
// ring communication. The testing set is kept local, and as a result the
// output priority queue is kept intact
//
void KNN::train() {
    // Perform first data exchange

    size_t send_samples_size = trainset->samples_size, recv_samples_size = 0;
    size_t send_row_ptr_size = trainset->row_ptr_size, recv_row_ptr_size = 0;

    double *comp_samples_buf = NULL; 
    size_t *comp_rptr_buf = NULL;

    Distance dist;
    
    for (int i = 0; i < comm->size(); i++) {
        // initiate the exchange of send and receive buffers
       initiate_send_and_recv(i, send_row_ptr_size, send_samples_size,
               recv_row_ptr_size, recv_samples_size, comp_samples_buf,
               comp_rptr_buf); 
        
       if (i & 1) { // odd
           comp_samples_buf = static_cast<double *>(& (trainset->samples[0]));
           comp_rptr_buf = static_cast<size_t*>(& (trainset->row_ptr[0]));
       }
       else { // even
           comp_samples_buf = static_cast<double*>(& (recv_samples[0]));
           comp_rptr_buf = static_cast<size_t*>(& (recv_row_ptr[0])); 
       }
       for (size_t j = 0; j < testset->row_ptr_size; j++) {
            size_t firststart = j > 0 ? testset->row_ptr[j - 1] : 0;
            size_t firstlen = j > 0 ? testset->row_ptr[j] - testset->row_ptr[j - 1] : testset->row_ptr[j];
            
            // reduce the first length to eliminate the classvariable
            firstlen--;

            // increment the firststart variable to show the correct index
            firststart++;

            double *first = static_cast<double *>(&(testset->samples[firststart]));

            for (size_t k = 0; k < recv_row_ptr_size; k++) {
                size_t secondstart = k > 0 ? comp_rptr_buf[k - 1] : 0;
                size_t secondlen = k > 0 ? comp_rptr_buf[k] - comp_rptr_buf[k - 1] : comp_rptr_buf[k];  

                int secondclass = (int)comp_samples_buf[secondstart]; // First element is the class

                secondlen--; secondstart++;
                double *second = &(comp_samples_buf[secondstart]);

                // calculate the distance between first and second samples
                double firstseconddistance = dist.euclideansparse(first, firstlen, second, secondlen);

                // create an output element and insert it in the priority
                // queue for that point
                struct output_elem outelem;
                outelem.classvar = secondclass;
                outelem.distance = firstseconddistance;

                // insert
                pq[j].push(outelem);

                // remove enough elements so that the space complexity is
                // restricted
                while(pq[j].size() > K)
                    pq[j].pop();

            }

        } 

       // set the values for next exchange
       send_row_ptr_size = recv_row_ptr_size;
       send_samples_size = recv_samples_size;
    }

}

void KNN::test() {

    // map for maintaining the count of classes
    map<int, int> classmap;
    classmap.clear();

    // variable to store the classes generated for each point by knn
    vector<int> pointclass;
    pointclass.resize(testset->num_local_samples);

    int maxclassinstance = 0;

    for(size_t i = 0; i < testset->num_local_samples; ++i) {
        classmap.clear();
        maxclassinstance = 0;
        
        // deque from priority queue and count the instances of classes
        while(pq[i].size()) {
            struct output_elem elem;
            elem = pq[i].top();
            ++classmap[elem.classvar];

            if (classmap[elem.classvar] > maxclassinstance)
                maxclassinstance = elem.classvar;

            pq[i].pop();
        }
        pointclass[i] = maxclassinstance;
    }

    MPI_Barrier(comm->worldcomm());
       
    size_t missedclasses = 0; // a variable to store number of missed classes

    for (int rankiter = 0; rankiter < comm->size(); ++rankiter) {
        if ((rankiter + 1) % comm->size() == comm->rank()) { 
            for (size_t j = 0; j < testset->row_ptr_size; j++) {
                size_t firststart = j > 0 ? testset->row_ptr[j - 1] : 0;
                size_t firstlen = j > 0 ? testset->row_ptr[j] - testset->row_ptr[j - 1] : testset->row_ptr[j];
                int actualclass = testset->samples[firststart];        

                if ((int)actualclass != (int)pointclass[j])
                    ++missedclasses;

                // reduce the first length to eliminate the classvariable
                firstlen--;

                // increment the firststart variable to show the correct index
                firststart++;

                for(size_t k = 0; k < firstlen; k += 2) {
                }
            }
        }
        MPI_Barrier(comm->worldcomm());
    }

    int rc = MPI_Allreduce(MPI_IN_PLACE, &missedclasses, 1, MPI_LONG, MPI_SUM, comm->worldcomm());

    if (0 == comm->rank()) {
        cout << "Accuracy: " << 100 - (1.0* missedclasses / testset->numglobalsamples()) * 100 << "%" << endl;
    }

    MPI_Barrier(comm->worldcomm());
}

KNN::~KNN() {
    delete comm;
    delete trainset;
}
