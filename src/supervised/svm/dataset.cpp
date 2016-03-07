#include "dataset.hpp"
#include <climits>
#include <string>
#include <fstream>
#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <map>

using namespace std;

size_t Tokenize(string& str,
                      vector<double>& tokens,
                      const string& delimiters = " :,\t")
{
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    string::size_type pos     = str.find_first_of(delimiters,
            lastPos);

    size_t ntokens = 0;
    while (string::npos != pos || string::npos !=
            lastPos)
    {
        // Found a token, add it to the vector.
        string token = str.substr(lastPos,
                                    pos - lastPos);
        const char *tmp = token.c_str();
        double value = strtod(tmp, NULL);
        tokens.push_back(value);
        lastPos = str.find_first_not_of(delimiters,
                pos);
        pos = str.find_first_of(delimiters,
                lastPos);
        ++ntokens;
    }
    return ntokens;
}

size_t find_num_samples(ifstream &in) {
    size_t numLines = 0;
    vector<double> tmp_samples;
    while (in.good()){
        string line;
        getline(in, line);
        size_t ntokens = Tokenize(line, tmp_samples);

        if (ntokens)
            ++numLines;
    }
    return numLines;
}

Dataset::Dataset(char *f, int rank) {
    // Reading is done by process 0
    if (0 == rank) {
        // Check if filename exists
        if (!f) {
            cout << "Error, No Existing Filename" << endl;
            return;
        }

        ifstream file_ptr;
        file_ptr.open(f);

        // Check if the file actually exists
        if (!file_ptr.is_open()) {
            cout << "Error, File does not exist!!!" << endl;
            assert(0);
        }

        // Calculate the number of Lines
        num_global_samples = find_num_samples(file_ptr);
        file_ptr.close();

        // open the file with a new stream and save it
        inp_file.open(f);
    }

}

Dataset::~Dataset() {
    inp_file.close();
}

size_t Dataset::numglobalsamples() const {
    return num_global_samples;
}

size_t Dataset::numlocalsamples() const {
    return num_local_samples;
}

void Dataset::setnumglobalsamples(size_t num) {
    num_global_samples = num;
}

void Dataset::exchange_data(Comm *comm, size_t &sample_count, size_t &accum_row_ptr_value, int partner) {
    int rc= 0;

    if (comm->size() == 1) { // base case
        return;
    }

    void * data; // pointer for sending and receiving data
    if (comm->rank() == 0) {
        // send the size of the data to be sent
        rc = MPI_Send(&accum_row_ptr_value, sizeof(size_t), MPI_BYTE, partner, 0, comm->worldcomm());
        assert(0 == rc);

        // send size of row pointer
        rc = MPI_Send(&sample_count, sizeof(size_t), MPI_BYTE, partner, 0, comm->worldcomm());
        assert(0 == rc);
        
        data = static_cast<void*>(& samples[0]); 
        rc = MPI_Send(data, accum_row_ptr_value, MPI_DOUBLE, partner, 0, comm->worldcomm());
        assert(0 == rc);
    
        data = static_cast<void*>(& row_ptr[0]); 
        rc = MPI_Send(data, sample_count, MPI_DOUBLE, partner, 0, comm->worldcomm());
        assert(0 == rc);
        sample_count = accum_row_ptr_value = 0;
    }
    else {
        MPI_Status status;

        // resize the samples array
        rc = MPI_Recv(&accum_row_ptr_value, sizeof(size_t), MPI_BYTE, partner, 0, comm->worldcomm(), &status);
        assert(0 == rc);
        samples.resize(accum_row_ptr_value);

        // resize the row ptr
        rc = MPI_Recv(&sample_count, sizeof(size_t), MPI_BYTE, partner, 0, comm->worldcomm(), &status);
        assert(0 == rc);
        row_ptr.resize(sample_count);
   
        // receive the samples array
        data = static_cast<void*>(& samples[0]); 
        rc = MPI_Recv(data, accum_row_ptr_value, MPI_DOUBLE, partner, 0, comm->worldcomm(), &status);
        assert(0 == rc);
    
        data = static_cast<void*>(& row_ptr[0]); 
        rc = MPI_Recv(data, sample_count, MPI_DOUBLE, partner, 0, comm->worldcomm(), &status);
        assert(0 == rc);
    }
}

size_t Dataset::calculate_global_index(Comm *comm, size_t local_index) {
    return (ds_start_index + local_index);
}

int Dataset::get_rank_from_index(Comm *comm, size_t local_index) {
    size_t samples_per_proc = numglobalsamples() / comm->size();
  
    size_t elems_for_rank_0 = numglobalsamples() - samples_per_proc * (comm->size() - 1);
    
    if (local_index < elems_for_rank_0)
        return 0;

    // The remaining number should be divisible by samples per proc
    assert((numglobalsamples() - elems_for_rank_0) % samples_per_proc == 0);

    int rank = (1 + (local_index - elems_for_rank_0) / samples_per_proc);
    assert(rank < comm->size());
    return rank;
}

void Dataset::setindices(Comm *comm) {
    size_t samples_per_proc = numglobalsamples() / comm->size();
    if (comm->rank() == 0) {
        ds_start_index = 0;
        ds_end_index = numglobalsamples() - samples_per_proc * (comm->size() - 1);
    }
    else {
        ds_start_index = numglobalsamples() - samples_per_proc * (comm->size() - 1) + (comm->rank() - 1) * samples_per_proc;
        ds_end_index = ds_start_index + samples_per_proc; 
    }
}

void Dataset::munge(Comm *comm) {
    map<int, int> classmap;
    map <int, int>::iterator classmap_it;
    int classval[2] = {-1, 1};
    int classval_count = 0;
    int found_pone = 0, found_mone = 0;
    size_t pone_index = LONG_MAX, mone_index = LONG_MAX;

    for (size_t i = 0; i < row_ptr.size(); ++i) {
        size_t nelems, offset;
        if (0 == i) {
            nelems = row_ptr[i];
            offset = 0;
        } 
        else {
            offset = row_ptr[i - 1];
            nelems = row_ptr[i] - row_ptr[i - 1];
        }

        int classvar = samples[offset];
        assert(classvar == 1 || classvar == -1);
        
        for (size_t j = 0; j < nelems - 4; j++) {
           samples[offset + j] = samples[offset + j + 1]; 
        }

#if 0
        classmap_it = classmap.find(classvar);
        if (classmap_it == classmap.end() && classval_count < 2) {
            classvar = classval[classval_count];
            classval_count++;
        }
        else{
            if (classmap_it == classmap.end() && classval_count == 2) {
            //cout << "More than 2 classes" << endl;
            //exit (-1);
            }
        }
#endif   
       if (!found_pone && classvar == 1) {
           found_pone = 1;
           pone_index = calculate_global_index(comm, i);
       } 

       if (!found_mone && classvar == -1) {
            found_mone = 1;
            mone_index = calculate_global_index(comm, i);
       }

        samples[offset + nelems - 1] = classvar;
        samples[offset + nelems - 4] = 0.0; // alpha
        if (classvar == 1) {
            samples[offset + nelems - 3] = 1.0;
            samples[offset + nelems - 2] = -1.0;
        }
        if (classvar == -1) {
            samples[offset + nelems - 3] = 4.0;
            samples[offset + nelems - 2] = 1.0;

        }
    }

    MPI_Allreduce(&pone_index, &i_up, 1, MPI_LONG, MPI_MIN,  comm->worldcomm());
    MPI_Allreduce(&mone_index, &i_low, 1, MPI_LONG, MPI_MIN,  comm->worldcomm());
#if 0
    for (int i = 0; i < comm->size(); i++) {
        if (i == comm->rank()) {
            cout << "rank: " << comm->rank() << endl;;
            print(comm);
        }
        MPI_Barrier(comm->worldcomm());
    }
#endif
}


bool Dataset::readsparse(Comm *comm, size_t offset) {
  
    int size = comm->size();
    string line;

    // number of samples each process owns
    size_t samples_per_proc = num_global_samples / size;

    // counter for number of samples read up to now
    num_local_samples = 0;

    // accumulated value of row_ptr
    size_t accum_row_ptr_value = 0;

    max_sample_size = 0;

    if (comm->rank() == 0) {
        // start with dst = 1
        int dst = 1;
        while (inp_file.good()) {

            getline(inp_file, line);
            size_t ntokens = Tokenize(line, samples);
            // check for empty lines
            if (ntokens) {
                // For each sample keep adding the offset
                size_t cur_size = samples.size();    
                samples.resize(cur_size + offset);
                accum_row_ptr_value += ntokens + offset;
                row_ptr.push_back(accum_row_ptr_value);
                ++num_local_samples;
                if (max_sample_size < ntokens + offset)
                    max_sample_size = ntokens + offset;
            }    

            // if you have rotated to yourself, accumulate the dataset 
            // to yourself
            if (num_local_samples == samples_per_proc && dst != comm->rank() && comm->size() > 1) {
                if (dst != comm->rank()) {
                    exchange_data(comm, num_local_samples, accum_row_ptr_value, dst);
                    samples.resize(0);
                    row_ptr.resize(0);
                    assert(samples.empty() && row_ptr.empty());
                }
                dst++;
                dst = dst % comm->size(); 
            }
        }
    }
    else {
        exchange_data(comm, num_local_samples, accum_row_ptr_value, 0);
    }
#if 0
    for (int i = 0; i < comm->size(); i++) {
        if (i == comm->rank()) {
            cout << "rank: " << comm->rank() << endl;;
            print(comm);
        }
        MPI_Barrier(comm->worldcomm());
    }
#endif
    return true;
}


void Dataset::print(Comm *comm) {

    for (int i = 0; i < comm->size() ; i++ ) {

        if (i == comm->rank()) {
            cout << "rank: " << comm->rank() << endl;
            cout << "num global samples: " << num_global_samples << endl;
            cout << "num local samples: " << num_local_samples << endl;

            size_t rptr_count = 0;
            cout << "row_ptr:" << endl;
            for (size_t j = 0; j < row_ptr.size(); j++) {
                cout << row_ptr[j] << " ";
            }
            cout << endl;
            cout << "samples:" << endl;
            for (size_t j = 0; j < samples.size(); j++) {
                cout << samples[j] << " ";

                if (j == row_ptr[rptr_count] - 1) {
                    cout << endl;
                    ++rptr_count;
                }
            }
            cout << endl;
            cout << "i_up: " << i_up << ", i_low: "<< i_low <<  endl;
            cout <<"max_sample_size: " << max_sample_size << endl;
            cout <<"ds_start_index: " << ds_start_index << ", ds_end_index: " << ds_end_index <<  endl;
        }
        MPI_Barrier(comm->worldcomm());
    }


    vector<size_t>::iterator riter;
#if 0
    for (riter = row_ptr.begin(); riter != row_ptr.end(); ++riter) {
        cout << *riter << " ";
    }
    cout << endl;
#endif
}

Dataset::Dataset()
{
}
