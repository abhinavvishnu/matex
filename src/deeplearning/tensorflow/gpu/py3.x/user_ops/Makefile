CC=mpicc

ifeq ($(origin MTX_INSTALL_DIR), undefined)
	TF_INC=../matex_tf/lib/python3.5/site-packages/tensorflow/include
else
	TF_INC=$(MTX_INSTALL_DIR)/matex_tf/lib/python3.5/site-packages/tensorflow/include
endif

CFLAGS=-O2 -fPIC -shared -I $(TF_INC) -std=c++11 -msse -mavx -D_GLIBCXX_USE_CXX11_ABI=0
NVCC=$(CUDA_HOME)/bin/nvcc
CU_LIBS=$(CUDA_HOME)/lib:$(CUDA_HOME)/lib64
CU_INC=$(CUDA_HOME)/include

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
