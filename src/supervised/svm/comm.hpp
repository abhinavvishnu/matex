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
        int rank() const;
        int size() const;
        void print() const;
        double timestamp();
        MPI_Comm worldcomm() const;
};
