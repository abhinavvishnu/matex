CC=mpicc
TF_INC=../py_distro/lib/python3.4/site-packages/tensorflow/include/
CFLAGS=-O2 -fPIC -shared -I $(TF_INC) -std=c++11 -msse -mavx
#CUDA_HOME=/usr/local/cuda-6.5
NVCC=$(CUDA_HOME)/bin/nvcc
CU_LIBS=$(CUDA_HOME)/lib:$(CUDA_HOME)/lib64
CU_INC=$(CUDA_HOME)/include

#MPI_HOME=/share/apps/openmpi/1.8.3/gcc/4.9.2/
MPI_INC=$(MPI_HOME)/include
CUFLAGS=-O2 -shared -I $(CU_INC) -I $(TF_INC) -I $(MPI_INC) -std=c++11 --compiler-options '-fPIC'

all: tf_reduce.so tf_broadcast.so tf_bind.so tf_sync.so

tf_reduce.so: tf_reduce.cc
	$(CC) $(CFLAGS) -o tf_reduce.so tf_reduce.cc

tf_broadcast.so: tf_broadcast.cc
	$(CC) $(CFLAGS) -o tf_broadcast.so tf_broadcast.cc

tf_sync.so: tf_sync.cc
	$(CC) $(CFLAGS) -o tf_sync.so tf_sync.cc

tf_bind.so: tf_bind.cc
	$(NVCC) $(CUFLAGS) -o tf_bind.so tf_bind.cc

clean: 
	rm -rf *.so
