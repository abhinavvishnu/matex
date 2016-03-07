#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <string>
#include <limits>



#include "kmeans.hpp"

kmeans::kmeans() {
    comm = NULL;
    dataset = NULL;
    MAX_ITER = 5;
}

kmeans::kmeans(int *argc, char ***argv) {
    kmeans();
    
    char **tmp = *argv;
    assert(*argc == 3);
    comm = new Comm(argc, argv);
    dataset = new Dataset(tmp[1], comm->rank());
   
    // process 0 reads the number of samples 
    int root = 0;
    size_t n_samples;
    
    if (0 == comm->rank())
        n_samples = dataset->getnumglobalsamples();

    MPI_Bcast(&n_samples, sizeof(size_t), MPI_BYTE, root,  comm->worldcomm());
    
    // set the correct values after broadcast
    dataset->setnumglobalsamples(n_samples);
    // read the file 
    if(dataset->readsparse(comm) == false) {
        cout << "Error in reading the file" << endl;
    }

    num_centroids = atol(tmp[2]);

    if (num_centroids <= 0)
        cout << "Invalid number of centroids" << endl;
}

size_t find_max_dimension(vector<double> &samples) {
    
    int maxdim = 0;
    vector<double>::iterator samp_iter;

    // every alternate index is a dimension
    for(samp_iter = samples.begin(); samp_iter!= samples.end(); samp_iter += 2) {
        maxdim = (maxdim > *samp_iter) ? maxdim : *samp_iter;
    }

    return maxdim;
}

size_t kmeans::pick_local_centroids(int me, int size) {

    if (num_centroids < (size_t) size) {
        if ((size_t) me < num_centroids)
            return 1;
        else
            return 0;
    }
    else {
        int div = num_centroids / size;
        int rem = num_centroids % size;

        if (me != size - 1) {
            return div;
        }
        else
            return (num_centroids - div * (size - 1));
    }
}

void kmeans::seed() {

    // Helper function to find the max number of dimensions
    // This function should not be a part of dataset class
    maxdim = find_max_dimension(dataset->samples);

    int allredrc = MPI_Allreduce(MPI_IN_PLACE, &maxdim, 1, MPI_LONG, MPI_MAX, comm->worldcomm());
    assert(allredrc == MPI_SUCCESS);

    // now resize the centroids and each of its dimension
    // flat vector for easier communication
    centroids.resize(num_centroids * maxdim);

    // pick the number of initial centroids
    size_t num_local_centroids = pick_local_centroids(comm->rank(), comm->size());

    double *samples = static_cast<double*>(& (dataset->samples[0]));
    size_t *row_ptr = static_cast<size_t*>(& (dataset->row_ptr[0]));

    // reset each element of centroids
    for (size_t i = 0 ; i < centroids.size(); i++)
        centroids[i] = 0;

    vector<double> local_centroids;
    local_centroids.resize(num_local_centroids * maxdim);

    // fill in the local values for centroids
    for (size_t i = 0; i < num_local_centroids; i++) {
        size_t sample_size = (i == 0) ? row_ptr[i] : row_ptr[i] - row_ptr[i - 1];
        size_t offset = (i == 0) ? 0  : row_ptr[i - 1];

        for (size_t j = 0; j < sample_size; j += 2 ) {
            double col = samples[offset + j];
            double value = samples[offset + j + 1];
            local_centroids[i * maxdim + col - 1] = value;
        }
    }

    vector<int> displs, recvcount;
    displs.resize(comm->size());
    recvcount.resize(comm->size());

    displs[0] = 0; // default
    recvcount[0] = pick_local_centroids(0 , comm->size()) * maxdim;

    //cout << displs[0] << " " << recvcount[0] << endl;
    // fill in all the displacements
    for (int i = 1; i < comm->size(); i++) {
        displs[i] = displs[i - 1] + recvcount[i -1];
        recvcount[i] = pick_local_centroids(i , comm->size()) * maxdim;
      //  cout << displs[i] << " " << recvcount[i] << endl;
    }
    double * lcen = static_cast<double*>(& local_centroids[0]);
    double * gcen = static_cast<double*>(& centroids[0]);
    
#if MPI_VERSION >= 3
    const int *ptr_displs = static_cast<int*>(& displs[0]);
    const int *ptr_recvcount = static_cast<int*>(& recvcount[0]);
#else
    int *ptr_displs = static_cast<int*>(& displs[0]);
    int *ptr_recvcount = static_cast<int*>(& recvcount[0]);
#endif
    int rc = MPI_Allgatherv(lcen, num_local_centroids * maxdim, MPI_DOUBLE, gcen, ptr_recvcount, 
            ptr_displs, MPI_DOUBLE, comm->worldcomm());

    assert(MPI_SUCCESS == rc);

#if 0
    for (int j = 0 ; j < comm->size(); j++) {
        if (comm->rank() == j) {
            for (size_t i = 0; i < num_centroids; i++) {
                for (size_t j = 0; j < maxdim; j++) {
                    cout << centroids[i * maxdim + j] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
        MPI_Barrier(comm->worldcomm());
    }
#endif
}

// inline helper function
static inline void convert_sparse_to_dense(vector<double> &samples,
        vector<size_t> &row_ptr, size_t index, vector<double> &dense_sample){

    size_t start_index, len;
    if (!index) {
        start_index = 0;
        len = row_ptr[index];
    }
    else {
        start_index = row_ptr[index - 1];
        len = row_ptr[index] - row_ptr[index - 1];
    }
    for (size_t i = 0; i < len; i += 2) {
        double col = samples[start_index + i] - 1;
        double val = samples[start_index + i + 1];
        dense_sample[col] = val;
    }
#if 0
    for(size_t i = 0; i < dense_sample.size(); ++i)
        cout << dense_sample[i] << " ";

    cout << endl;
#endif
}


void kmeans::iterative() {

    // create a vector which keeps running copy
    vector<double> running_centroids;
    running_centroids.resize(maxdim * num_centroids);

    // reset to 0
    fill(running_centroids.begin(), running_centroids.end(), 0);

    // create a vector to keep track of number of samples in each centroid
    vector<size_t> num_samples_in_centroid;
    num_samples_in_centroid.resize(num_centroids);

    //reset to 0
    fill(num_samples_in_centroid.begin(), num_samples_in_centroid.end(), 0);
    
    vector<double> dense_sample;
    dense_sample.resize(maxdim);

    Distance dist;

    vector<size_t> point_centroid;
    
    point_centroid.resize(dataset->getnumlocalsamples());
    fill(point_centroid.begin(), point_centroid.end(), numeric_limits<double>::max());

    size_t num_changed = 0, iter_count = 0;

    do{
        num_changed = 0;
        for (size_t i = 0; i < dataset->getnumlocalsamples(); i++) {


            // convert the sparse sample -> dense sample
            fill(dense_sample.begin(), dense_sample.end(), 0);
            convert_sparse_to_dense(dataset->samples, dataset->row_ptr, i, dense_sample);

            double min_distance = numeric_limits<double>::max(); 
            double running_distance;
            size_t min_centroid;

            // calculate the distance with each centroid
            for (size_t j = 0; j < num_centroids; j++) {
                running_distance = dist.euclidean(static_cast<double *>( &dense_sample[0]),
                        static_cast<double *> (&(centroids[j * maxdim])), maxdim);  
                // update the best centroid
               
                if (running_distance < min_distance) {
                    min_centroid = j;
                    min_distance = running_distance;
                }
            }

            if (point_centroid[i] != min_centroid) {
                point_centroid[i] = min_centroid;
                ++num_changed;
            }
            // update the number of samples in the centroid
            ++num_samples_in_centroid[min_centroid];

            // update the centroid themselves 
            dist.addvectors(static_cast<double*> (&dense_sample[0]),
                    static_cast<double *> (&running_centroids[min_centroid *
                        maxdim]), maxdim);
        }
        // Allreduce the number of samples in each centroid
        
        MPI_Allreduce(MPI_IN_PLACE, &num_changed, 1, MPI_LONG, MPI_SUM, comm->worldcomm());
        
        MPI_Allreduce(MPI_IN_PLACE, static_cast<size_t *>(&num_samples_in_centroid[0]), num_centroids,
                    MPI_LONG, MPI_SUM, comm->worldcomm());
        
        fill(centroids.begin(), centroids.end(), 0);
        // Allreduce the centroids samples
        MPI_Allreduce(static_cast<double*>(&running_centroids[0]), static_cast<double*>(&centroids[0]),
                maxdim * num_centroids, MPI_DOUBLE, MPI_SUM, comm->worldcomm());

        // Divide the centroids with their respective number of samples
       
        dist.divvectors(static_cast<double*>(&centroids[0]), static_cast<size_t*>(&num_samples_in_centroid[0]), 
               num_centroids, maxdim); 
        // Clear the contents
        fill(num_samples_in_centroid.begin(), num_samples_in_centroid.end(), 0);
        fill(running_centroids.begin(), running_centroids.end(), 0);
        ++iter_count;

    } while (iter_count < 20 && num_changed);

    print_stats();
}

void kmeans:: print_stats() {
    if (0 == comm->rank()) {
        cout << "Printing the centroids:" << endl;
        for (size_t i = 0; i < num_centroids; i++) {
            for (size_t j = 0; j < maxdim; j++) {
                if (centroids[i * maxdim + j]) 
                    cout << j  + 1 << ":" << centroids[i * maxdim + j] << " ";
            }
            cout << endl;
        }
    }
    MPI_Barrier(comm->worldcomm());
}

kmeans::~kmeans() {
    delete comm;
    delete dataset;
}
