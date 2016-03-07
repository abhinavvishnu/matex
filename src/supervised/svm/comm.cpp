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
    int rc;
     rc = MPI_Init(argc, argv);
    assert(!rc);
    
    //comex_init_args(argc, argv);
    //MPI_Comm comex_comm;

    //comex_group_comm(COMEX_GROUP_WORLD, &comex_comm);

    rc = MPI_Comm_dup(MPI_COMM_WORLD, &world_comm);
    assert(!rc);

    rc = MPI_Comm_size(world_comm, &num_proc);
    assert(!rc);

    rc = MPI_Comm_rank(world_comm, &my_rank);
    assert(!rc);
    
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
