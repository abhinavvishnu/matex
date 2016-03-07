#include <iostream>
#include <mpi.h>

class Comm {
    private:
        int my_rank;
        int num_proc;
        MPI_Comm world_comm;
    public:
        Comm();
        Comm(int *, char ***);
        ~Comm();
        const int rank() const;
        const int size() const;
        void print() const;
        const MPI_Comm worldcomm() const;
};
