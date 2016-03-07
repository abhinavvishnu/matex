#include "comm.hpp"
#include <mpi.h>
#include <assert.h>

using namespace std;

Comm::Comm()
{
    my_rank = 0;
    num_proc = 0;
}

Comm::Comm(int *argc, char ***argv) {
    int rc = MPI_Init(argc, argv);
    assert(!rc);

    rc = MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    assert(!rc);

    rc = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    assert(!rc);
    
    rc = MPI_Comm_dup(MPI_COMM_WORLD, &world_comm);
    assert(!rc);
}

const int Comm::rank() const {
    return my_rank;
}

const int Comm::size() const {
    return num_proc;
}

const MPI_Comm Comm::worldcomm() const {
    return world_comm;
}

void Comm::print() const{
    cout << "My rank: " << my_rank << endl;
    cout << "Total Process Count: " << num_proc << endl;
}

Comm::~Comm() {
    MPI_Finalize();
}
