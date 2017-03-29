MATEX Distributed TensorFlow Version 0.1
========================================

Deep Learning (DL) algorithms have become the de facto choice for data analysis. 
Several DL implementations -- primarily limited to a single compute node -- 
such as Caffe, TensorFlow, Theano and Torch have become readily available. 
Distributed DL implementations capable of execution on large
scale systems are becoming important to address the computational needs of
large data produced by scientific simulations and experiments.  Yet, the
adoption of distributed DL faces significant impediments: 1) Most
implementations require DL analysts to modify their code significantly -- which
is a show-stopper, 2) Several distributed DL implementations are
geared towards cloud computing systems -- which is inadequate for execution on
massively paper supercomputers.
 
This software release aims to alleviate the transition between single node DL
frameworks and full distributed high performance DL frameworks. We provide a 
distributed memory DL implementation by incorporating required changes in 
the TensorFlow runtime itself. This dramatically reduces the entry barrier 
for using distributed TensorFlow implementation.  We use Message Passing 
Interface (MPI) -- which provides performance portability, especially 
since MPI specific changes are abstracted from users. Lastly -- and arguably 
most importantly -- we make our implementation available for broader use, 
under the umbrella of Machine Learning Toolkit for Extreme Scale (MaTEx) 
at http://hpc.pnl.gov/matex.

These git folder has all the requirements to install and run the alpha release 
of MPI enabled tensorflow. This version is based on the 1.0.0 release of 
tensorflow. It is based on modified parts of the back end of tensorflow with 
MPI enabled operators and it will transparently parallelize the tensorflow 
applications, as long as the data is initialize partitioned across the 
computational resources. This packages contains wheels for both CPU and GPU 
versions.

Supported OSes and Environments
-------------------------------

The following has been tested in:

OSes:
- RedHat 6 
- Ubuntu 16
- Mint Linux

Hardware Configurations:
- X86-64 Clusters using Infiniband QDR
- GPUs Based clusters using K40


Installation
-------------

Please refer to INSTALL.txt for a complete set of instructions.


Constraints and Known Bugs
--------------------------

- The scripts will try to create an python virtual environment. Please do not 
try to install this inside another virtual environment
- Because of its alpha release, the operators will not handle multiple ranks in a single node with a single GPU, neither multiple parallel optimizer calls
- To ensure that the operator is called, you need to use the tensorflow optimize or compute_gradients calls in your model
- To disable the extensions, please set the TF_MPI_ENABLE to 0 and to renable them set it to 1
- Full Keras integration is on the works
- Mapping of multiple GPUs and Ranks is forthcoming
- For OpenMPI: the extensions do not behave well with EPOLL mode (known OpenMPI bug when redirecting output). Please use the poll MCA OPAL option:  --mca opal_event_include poll

