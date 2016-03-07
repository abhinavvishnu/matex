#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <string>
#include <limits>
#include <map>
#include <algorithm>
#include "fpgrowth.hpp"

using namespace std;

FPG::FPG() {
    comm = NULL;
    dataset = NULL;
}

FPG::FPG(int *argc, char ***argv) {
    char **tmp = *argv;
    comm = NULL;
    dataset = NULL;
    
    extra_elems = 0;
    ROOT = -2;
    delim = -1;
    max_sample_size = 256;

    assert(*argc == 3); 

    comm = new Comm(argc, argv);

    dataset = new Dataset(tmp[1], comm->rank());

    SUPPORT_COUNT = atof(tmp[2]);
    // process 0 reads the number of samples 
    int root = 0;
    size_t dataset_samples;
    
    if (0 == comm->rank()) {
        dataset_samples = dataset->numglobalsamples();
    }

    int rc;
    rc = MPI_Bcast(&dataset_samples, sizeof(size_t), MPI_BYTE, root,  comm->worldcomm());
    assert(rc == MPI_SUCCESS);

    // set the correct values after broadcast
    dataset->setnumglobalsamples(dataset_samples);
     
    // read the training file and allow extra elems to be appended on each sample 
    if(dataset->readsparse(comm, extra_elems, max_item_id) == false) {
        cout << "Error in reading the training set file" << endl;
    }

    // cache the values
    dataset->row_ptr_size = dataset->row_ptr.size();
    dataset->samples_size = dataset->samples.size();
    
    //dataset->print(comm);

    rc = MPI_Bcast(&max_item_id, 1, MPI_LONG, 0, comm->worldcomm());
    assert(rc == MPI_SUCCESS);

    item_freq_count.resize(max_item_id + 1);

    MPI_Barrier(comm->worldcomm());
}

void FPG::find_frequent_ones() {
    for(size_t i = 0; i < dataset->samples.size(); i++) {
        ++item_freq_count[dataset->samples[i]];
    }
    size_t *item_freq_buffer = static_cast<size_t *>(&item_freq_count[0]);
   
    int rc = MPI_Allreduce(MPI_IN_PLACE, item_freq_buffer,  max_item_id + 1, MPI_LONG, MPI_SUM,  comm->worldcomm());
    assert(rc == MPI_SUCCESS);

    //dataset->print(comm);

    for(size_t i = 0; i < item_freq_count.size(); i++) {
        double item_sup_count = (item_freq_count[i] *1.0) / dataset->numglobalsamples();
        //double item_sup_count = item_freq_count[i];

        if (item_sup_count > SUPPORT_COUNT) {
            struct freq_item_list_t f;
            f.item_id = i;
            f.support = item_sup_count; 
            freq_itemid_list.push_back(f);
        }
    }
    std::sort(freq_itemid_list.begin(), freq_itemid_list.end(), compare_by_support());

    for (size_t i = 0; i < freq_itemid_list.size(); ++i) {
        freq_item_rank[freq_itemid_list[i].item_id] = i + 2;
#if 0
        cout << freq_itemid_list[i].item_id << " ";
#endif
    }
    // Put the rank of the root to be last
    freq_item_rank[ROOT] = 1;
}

void FPG::sort_sample(size_t offset, size_t nelems, double *sample_copy, size_t &num_freq_items_in_trans) {

    num_freq_items_in_trans = 0;

    for(size_t i = 0; i < nelems; i++) {
        // check if item is frequent
        double item_id = dataset->samples[offset + i];
        assert(item_id <= max_item_id);

        if (freq_item_rank[item_id]) {
            sample_copy[num_freq_items_in_trans++] = item_id;
        }
    }

    // if the sample contains no frequent item, this sample is a no-op
    if (!num_freq_items_in_trans)
        return;

    // sort
    for (size_t i = 0; i < num_freq_items_in_trans; i++) {
        for (size_t j = i + 1; j < num_freq_items_in_trans; j++) {
            if (freq_item_rank[sample_copy[i]] < freq_item_rank[sample_copy[j]]) {
                double inter_elem = sample_copy[i];
                sample_copy[i] = sample_copy[j];
                sample_copy[j] = inter_elem;        
            }
        }
    }

    // add delimiters
    size_t start = num_freq_items_in_trans;
    for (size_t i = 0; i < num_freq_items_in_trans; ++i) {
        sample_copy[start + i] = delim;
    }

    num_freq_items_in_trans *= 2;
}


void FPG::build_local_prefix_tree() {
    vector<double> sample_copy;
    vector<size_t> freq_copy;

    // resize the samples copy and freq copy
    sample_copy.resize(max_sample_size);
    freq_copy.resize(max_sample_size);

    // fill the first index of sample with ROOT
    sample_copy[0] = ROOT;

    // fill all the contents of freq copy to 1
    std::fill(freq_copy.begin(), freq_copy.end(), 1);

    // aggregate of number of global frequent items
    size_t num_global_freq_items = 0;
    for(size_t i = 0; i < freq_itemid_list.size(); ++i) {
        num_global_freq_items += freq_itemid_list[i].support * dataset->numglobalsamples();
    }

    num_global_freq_items *= 3;
    // resize the local trees accordingly
    local_freq_tree_a.resize(num_global_freq_items);
    local_freq_tree_b.resize(num_global_freq_items);
    local_item_tree_a.resize(num_global_freq_items );
    local_item_tree_b.resize(num_global_freq_items);

    cout << local_freq_tree_a.size() << endl;

    size_t local_item_tree_a_size = 0;
    size_t local_item_tree_b_size = 0;

    size_t op_tree_size = 0, input_tree_size = 0;

    bool alt_flag = true;
    for (size_t i = 0; i < dataset->row_ptr.size(); ++i) {
        // number of elements in this sample
        size_t nelems = i > 0 ? dataset->row_ptr[i] - dataset->row_ptr[i - 1] :  dataset->row_ptr[i];
       
        assert(nelems < max_sample_size / 2);
        // offset from the beginning
        size_t offset = i > 0 ? dataset->row_ptr[i - 1]: 0;
        
        size_t num_freq_items_in_trans = 0;
        sort_sample(offset, nelems, static_cast<double *>(&(sample_copy[1])), num_freq_items_in_trans); 

        if (!num_freq_items_in_trans)
            continue;
#if 0
        for (size_t j = 0; j < num_freq_items_in_trans + 1; ++j) {
            cout << sample_copy[j] << "  ";
        }
        cout << endl;
#endif
        if (i % 100 == 0)
            cout << "Completed " << i << " samples" << endl;

        op_tree_size = input_tree_size + num_freq_items_in_trans + 1;

        if (alt_flag) {
            op_tree_size = fpmerge(sample_copy, 
                    (num_freq_items_in_trans + 1), freq_copy,
                    local_item_tree_a, input_tree_size,
                    local_freq_tree_a, local_item_tree_b, local_freq_tree_b,
                    freq_item_rank); 
            alt_flag = false;
        }
        else {
            op_tree_size = fpmerge(sample_copy, 
                    (num_freq_items_in_trans + 1), freq_copy,
                    local_item_tree_b, input_tree_size,
                    local_freq_tree_b, local_item_tree_a, local_freq_tree_a,
                    freq_item_rank);
           alt_flag = true; 
        }


        assert(op_tree_size <= local_item_tree_b.size());

        input_tree_size = op_tree_size;
    }

    local_prefix_tree_size = op_tree_size;

    MPI_Barrier(comm->worldcomm());
    if (alt_flag) {
        local_item_tree_a.resize(op_tree_size);
        local_freq_tree_a.resize(op_tree_size);
        local_item_tree_b.resize(0);
        local_freq_tree_b.resize(0);
    }
    else {
        local_item_tree_a.resize(0);
        local_freq_tree_a.resize(0);
        local_item_tree_b.resize(op_tree_size);
        local_freq_tree_b.resize(op_tree_size);
    }
}

void swap_buffer_pointers(vector<double>&send_item_vector, vector<double>&recv_item_vector,
        vector<size_t>&send_freq_vector, vector<size_t>&recv_freq_vector, double **send_item_buffer, double **recv_item_buffer,
        size_t **send_freq_buffer, size_t **recv_freq_buffer, int i) {


    if (i & 1) {
        *send_item_buffer = static_cast<double *>(&send_item_vector[0]);
        *send_freq_buffer = static_cast<size_t *>(&send_freq_vector[0]);
        *recv_item_buffer = static_cast<double *>(&recv_item_vector[0]);
        *recv_freq_buffer = static_cast<size_t *>(&recv_freq_vector[0]); 
    }
    else {
        *send_item_buffer = static_cast<double *>(&recv_item_vector[0]);
        *send_freq_buffer = static_cast<size_t *>(&recv_freq_vector[0]);
        *recv_item_buffer = static_cast<double *>(&send_item_vector[0]);
        *recv_freq_buffer = static_cast<size_t *>(&send_freq_vector[0]); 
    }
}

void FPG:: build_global_prefix_tree() {
    // Find the max size of the tree
    size_t max_tree_size = 0;

    int rc;
    rc = MPI_Allreduce(&local_prefix_tree_size, &max_tree_size, 1, MPI_LONG, MPI_MAX, comm->worldcomm());
    assert(rc == MPI_SUCCESS);

    // src and destination
    int src = (comm->rank() + comm->size() - 1) % comm->size();
    int dst = (comm->rank() + 1) % comm->size();

    // create two vectors each for send and receive and resize them to max
    // tree size
    vector<double> recv_item_vector, send_item_vector;
    vector<size_t> recv_freq_vector, send_freq_vector;
    
    recv_item_vector.resize(max_tree_size); send_item_vector.resize(max_tree_size);
    recv_freq_vector.resize(max_tree_size); send_freq_vector.resize(max_tree_size);

    double *send_item_buffer, *recv_item_buffer;
    size_t *send_freq_buffer, *recv_freq_buffer;
    size_t send_item_buffer_size, send_freq_buffer_size;

    // copy the original tree to the send buffer
    if (local_item_tree_b.size()){
        for (size_t i = 0; i < local_item_tree_b.size(); ++i) {
            send_item_vector[i] = local_item_tree_b[i];
            send_freq_vector[i] = local_freq_tree_b[i];
        }
        send_item_buffer_size = send_freq_buffer_size = local_item_tree_b.size();
    }
    else {
        for (size_t i = 0; i < local_item_tree_a.size(); ++i) {
            send_item_vector[i] = local_item_tree_a[i];
            send_freq_vector[i] = local_freq_tree_a[i];
        }
        send_item_buffer_size = send_freq_buffer_size = local_item_tree_a.size();
    }
    
    // Cast the send buffers
    send_item_buffer = static_cast<double *>(&send_item_vector[0]);
    send_freq_buffer = static_cast<size_t *>(&send_freq_vector[0]);

    // cast the receive buffers
    recv_item_buffer = static_cast<double *>(&recv_item_vector[0]);
    recv_freq_buffer = static_cast<size_t *>(&recv_freq_vector[0]); 
   
    // For sanity, clear the contents of a and b buffers and resize them to 
    // max_tree_size * comm->rank() : max tree size that will every grow
    // locally 
    

    local_item_tree_a.resize(0); local_item_tree_b.resize(0);
    local_freq_tree_a.resize(0); local_freq_tree_b.resize(0);

    local_item_tree_a.resize(max_tree_size * comm->size());
    local_item_tree_b.resize(max_tree_size * comm->size());
    local_freq_tree_a.resize(max_tree_size * comm->size());
    local_freq_tree_b.resize(max_tree_size * comm->size());

    MPI_Request req[4];
    MPI_Status status[4];

    size_t input_tree_size = 0, op_tree_size = 0;

    bool odd_flag = true;
    // Exchange send and receive buffers and merge them using the fpmerge
    // function
    for (int i = 0; i < comm->size(); ++i) {
        rc = MPI_Irecv(recv_item_buffer, max_tree_size, MPI_DOUBLE, src, 0, comm->worldcomm(), &(req[0])); 
        assert(rc == MPI_SUCCESS);

        rc = MPI_Irecv(recv_freq_buffer, max_tree_size, MPI_LONG, src, 1, comm->worldcomm(), &(req[1]));
        assert(rc == MPI_SUCCESS);

        rc = MPI_Isend(send_item_buffer, send_item_buffer_size, MPI_DOUBLE, dst, 0, comm->worldcomm(), &(req[2]));
        assert(rc == MPI_SUCCESS);
        
        rc = MPI_Isend(send_freq_buffer, send_freq_buffer_size, MPI_LONG, dst, 1, comm->worldcomm(), &(req[3]));
        assert(rc == MPI_SUCCESS);
        
        rc = MPI_Waitall(4, req, status);
        assert(rc == MPI_SUCCESS);

        int recv_item_count, recv_freq_count;
        rc = MPI_Get_count(&(status[0]), MPI_DOUBLE, &recv_item_count);
        rc = MPI_Get_count(&(status[1]), MPI_LONG, &recv_freq_count);
        assert(recv_freq_count == recv_item_count);
    
        send_item_buffer_size = send_freq_buffer_size = (size_t) recv_item_count;
        if (i & 1) {
            op_tree_size = fpmerge(send_item_vector, 
                    recv_item_count, send_freq_vector,
                    local_item_tree_a, input_tree_size,
                    local_freq_tree_a, local_item_tree_b, local_freq_tree_b,
                    freq_item_rank); 
            odd_flag = true;
        }
        else {
            op_tree_size = fpmerge(recv_item_vector, 
                    recv_item_count, recv_freq_vector,
                    local_item_tree_b, input_tree_size,
                    local_freq_tree_b, local_item_tree_a, local_freq_tree_a,
                    freq_item_rank);
            odd_flag = false;
        }
        assert(op_tree_size <= local_item_tree_a.size()); 

        swap_buffer_pointers(send_item_vector, recv_item_vector, send_freq_vector, recv_freq_vector, 
                &send_item_buffer, &recv_item_buffer, &send_freq_buffer, &recv_freq_buffer, i);

        input_tree_size = op_tree_size;
    }

    // synhronize
    MPI_Barrier(comm->worldcomm());

    if (0 == comm->rank()) {
        if (odd_flag) {
            for (size_t i = 0; i < op_tree_size; i++)
            cout << local_item_tree_b[i] << " ";
        }
        else {
            for (size_t i = 0; i < op_tree_size; i++)
            cout << local_item_tree_a[i] << " ";
        }
        cout << endl;
#if 0 
        for (size_t i = 0; i < op_tree_size; i++)
            cout << local_freq_tree_a[i] << " ";
        cout << endl;
#endif
    }
    MPI_Barrier(comm->worldcomm());

}


void FPG::run() {

    // find the local, global frequency of each item
    // and create the following arrays:
    // map for rank order of different items
    // sorted list of frequent items 
    find_frequent_ones();

    // build local prefix tree
    build_local_prefix_tree();

    // build global prefix tree
    build_global_prefix_tree();
}


void FPG::usage() {
    if (comm->rank() == 1) {
        cout << "Usage :: mpirun -np ./matex_fpgrowth proc_count dataset" << endl;
    }
}


FPG::~FPG() {
    delete comm;
    delete dataset;
}
