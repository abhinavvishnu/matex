CC=mpicc
TF_INC=../py_distro/lib/python3.4/site-packages/tensorflow/include/
CFLAGS=-O2 -fPIC -shared -I $(TF_INC) -std=c++11 -msse -mavx

all: tf_reduce.so tf_broadcast.so tf_sync.so

tf_reduce.so: tf_reduce.cc
	$(CC) $(CFLAGS) -o tf_reduce.so tf_reduce.cc

tf_broadcast.so: tf_broadcast.cc
	$(CC) $(CFLAGS) -o tf_broadcast.so tf_broadcast.cc

tf_sync.so: tf_sync.cc
	$(CC) $(CFLAGS) -o tf_sync.so tf_sync.cc

clean: 
	rm -rf *.so
