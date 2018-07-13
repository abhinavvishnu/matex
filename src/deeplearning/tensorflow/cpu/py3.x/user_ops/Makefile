CC=mpicc

ifeq ($(origin MTX_INSTALL_DIR), undefined)
	TF_INC=../matex_tf/lib/python3.5/site-packages/tensorflow/include
else
	TF_INC=$(MTX_INSTALL_DIR)/matex_tf/lib/python3.5/site-packages/tensorflow/include
endif

CFLAGS=-O2 -fPIC -shared -I $(TF_INC) -std=c++11 -msse -mavx -DCPU -D_GLIBCXX_USE_CXX11_ABI=0

all: tf_reduce.so tf_broadcast.so tf_sync.so tf_bind.so

tf_reduce.so: tf_reduce.cc
	$(CC) $(CFLAGS) -o tf_reduce.so tf_reduce.cc

tf_broadcast.so: tf_broadcast.cc
	$(CC) $(CFLAGS) -o tf_broadcast.so tf_broadcast.cc

tf_sync.so: tf_sync.cc
	$(CC) $(CFLAGS) -o tf_sync.so tf_sync.cc

tf_bind.so: tf_bind.cc
	$(CC) $(CFLAGS) -o tf_bind.so tf_bind.cc

clean:
	rm -rf *.so
