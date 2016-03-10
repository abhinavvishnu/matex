#include "comm.hpp"
#include <mpi.h>
#include <assert.h>

using namespace std;


Comm::Comm()
{
    my_rank = 0;
    num_proc = 0;
}

void mpi_operation(double *in, double *out, int *len, MPI_Datatype *type) {
    // the organization is bup, b_low, i_up, i_low
    assert(*len == 4);
    if (in[0] - out[0] < 1e-10) {
        out[0] = in[0];
        out[2] = in[2]; 
    }

    if (in[1] - out[1] > 1e-10) {
        out[1] = in[1];
        out[3] = in[3];
    }
}

Comm::Comm(int *argc, char ***argv) {
    int rc;
     rc = MPI_Init(argc, argv);
    assert(!rc);
    
    rc = MPI_Comm_dup(MPI_COMM_WORLD, &world_comm);
    assert(!rc);

    rc = MPI_Comm_size(world_comm, &num_proc);
    assert(!rc);

    rc = MPI_Comm_rank(world_comm, &my_rank);
    assert(!rc);
    
    // create the mpi operation

    rc = MPI_Op_create((MPI_User_function *)mpi_operation, 1, &mpi_op);
    assert(rc == MPI_SUCCESS);
}

int Comm::rank() const {
    return my_rank;
}

int Comm::size() const {
    return num_proc;
}

double Comm::timestamp() {
    return MPI_Wtime();
}


MPI_Comm Comm::worldcomm() const {
    return world_comm;
}

void Comm::print() const{
    cout << "My rank: " << my_rank << endl;
    cout << "Total Process Count: " << num_proc << endl;
}

Comm::~Comm() {
    MPI_Finalize();
}
