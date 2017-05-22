#include <stdio.h>
#include <mpi.h>

#ifndef CPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <iostream>
#include <unistd.h>

using namespace std;

char init = 0;

#include <thread>
#include <mutex>

extern "C" {
long *getAllRankIds(MPI_Comm comm, int size, int me);
int getGpuIDInNodeperRank(MPI_Comm comm, long *hostIds, int me);
int getGpuNumPerNode(MPI_Comm comm, long *hostIds, int me, int size);

void tfBind2GPU(){
#ifndef CPU
   int count = 0;
   int node_rank = 0;
   int comm_size = 0;
   int comm_rank = 0;
   int node_size = 0;
   MPI_Comm comm = MPI_COMM_WORLD;

   if(init == 1) return;

   cudaGetDeviceCount(&count);
   MPI_Comm_size(comm, &comm_size);
   MPI_Comm_rank(comm, &comm_rank);

   printf ("Size: %d Rank: %d\n", comm_size, comm_rank);

   long *hostIds = getAllRankIds(comm, comm_size, comm_rank);
   node_rank = getGpuIDInNodeperRank(comm, hostIds, comm_rank);
   node_size = getGpuNumPerNode(comm, hostIds, comm_rank, comm_size);

   if(node_size <= count){
      if(count != node_size){
         cout << "Warning MPI node size < cudaGetDeviceCount :" << node_size << " < " << count << endl;
      }
      cudaSetDevice(node_rank);
   }
   else{
      cerr << "Too many MPI Ranks per node" << endl;
      exit(11);
   }
   printf ("rank %d with %d gpus\n", node_rank, node_size);
#endif
   init = 1;
}


int tf_my_gpu(){
   int count = 0;
   int node_rank = 0;
   int comm_size = 0;
   int comm_rank = 0;
   int node_size = 0;
   MPI_Comm comm = MPI_COMM_WORLD;

   #ifdef CPU
      return -1;
   #else
   if(init == 1) return -1;

   cudaGetDeviceCount(&count);
   MPI_Comm_size(comm, &comm_size);
   MPI_Comm_rank(comm, &comm_rank);

   printf ("Size: %d Rank: %d\n", comm_size, comm_rank);

   long *hostIds = getAllRankIds(comm, comm_size, comm_rank);
   node_rank = getGpuIDInNodeperRank(comm, hostIds, comm_rank);
   node_size = getGpuNumPerNode(comm, hostIds, comm_rank, comm_size);

   if(node_size <= count){
      if(count != node_size){
         cout << "Warning MPI node size < cudaGetDeviceCount :" << node_size << " < " << count << endl;
      }
   }
   else{
      cerr << "Too many MPI Ranks per node" << endl;
      exit(11);
   }
   printf ("rank %d with %d gpus\n", node_rank, node_size);

   init = 1;
   return node_rank ;
   #endif
}



long *getAllRankIds(MPI_Comm comm, int size, int me)
{
#ifndef CPU
   if(size <= 0 || me < 0) return NULL;
   if(comm == MPI_COMM_NULL){
      comm = MPI_COMM_WORLD;
   }
   long *hostIds = new long[size];   
   hostIds[me] = gethostid();
   if(MPI_Allgather(MPI_IN_PLACE, 0, MPI_LONG, hostIds, 1, 
      MPI_LONG, comm) != MPI_SUCCESS){
      delete [] hostIds;
      cerr << "MPI Allgather failed" << endl;
      return NULL;
   }
   return hostIds; 
#else
   return NULL;
#endif
}
/* Util function Get the GPU related to the rank and node */
int getGpuIDInNodeperRank(MPI_Comm comm, long *hostIds, int me){
#ifndef CPU
   if(comm == MPI_COMM_NULL){
      comm = MPI_COMM_WORLD;
   }
   if(hostIds == NULL) return -1; 
   int i;
   int gpuId = 0;
   for(i = 0; i < me; ++i){
      if(hostIds[i] == hostIds[me]){
         gpuId ++;
      }
   }
   return gpuId; 
#else
   return -1;
#endif
}
int getGpuNumPerNode(MPI_Comm comm, long *hostIds, int me, int size){
#ifndef CPU
    if(comm == MPI_COMM_NULL){
       comm = MPI_COMM_WORLD;
    }
    if(hostIds == NULL) return -1;
    int i;
    int gpuSize = 0;
    for(i = 0; i < size; ++i){
       if(hostIds[i] == hostIds[me]){
          gpuSize ++;
      }
   }
   return gpuSize;
#else
   return -1;
#endif
}

}
