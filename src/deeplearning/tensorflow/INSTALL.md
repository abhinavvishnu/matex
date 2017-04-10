INSTALL for TF MPI X alpha
==========================

These folders have all the requirements to install and run the alpha release 
of MPI enabled tensorflow. This version is based on the 1.0.0 release of 
TensorFlow. It is based on an enhanced tensorflow back end with 
MPI enabled operators and it will transparently parallelize the tensorflow 
applications, as long as the data is initialize partitioned across the 
computational resources. This packages contains wheels for both CPU and GPU 
versions, as well as wheels for python 3.4 and 3.5 for x86_64.

The required software package are given below:

- GCC 4.9.2 or above
- OpenMPI 1.8.3 or above
- Python 3.4.2
- Autoconf, Make and m4
- Virtualenv and PIP 8 or above
- Access to the internet during the installation
- TCSH or Bash shell
- Linux Kernel above 3 **
- For GPU enabled clusters: CUDNN 4 and CUDA 7.5 

Please note that the kernel requirement is in asterick. That is because it is 
a tensorflow requirement that can be circumvented by our scripts and setup 
(more on that later).

Moreover, be aware that the CUDA environment when using an GPU cluster MUST
be functional. This means that the correct binaries and libraries are in the
correct place and pointed by the correct environment (i.e PATH and 
LD_LIBRARY_PATH).

Installation
------------

If you are using GPU enabled Tensorflow, you should set up the CUDNN_HOME 
environment variable to point where the CUDNN should be found (the head 
folder not the lib one).

Installing for bash shells

```
$ source ./install_mpi_tf.sh
```

Installing for C-shells

```
$ source ./install_mpi_tf.csh

```


Afterwards, you will be in a virtual python environment that 
encapsulates the tensorflow changes (Your 
shell prompt should look differently). 

```
[py_distro] $

```

[IMPORTANT:] A Note about older kernels
----------------------------------------

If you are using an older system (before Linux Kernel 3.0), TensorFlow 1.0 
might not even run on your system. We have provided a small workaround to do 
this and we created the setAlias.[csh|sh] scripts to help alleviate the 
transitions. 

For bash shells

```
source ./setAlias.sh

```

For C Shells

```
source ./setAlias.csh

```


This script sets the FAKE_SYSTEM_LIBS variable to point to where 
the updated versions of the system libraries are (provided by this package too)
and sets up an alias for a python enabled tensorflow (called pyflow) for 
interactive sessions. To run inside a script, please check the examples 
folders (py_scripts) to see how a single run would look like (it is not pretty 
but functional).

YOU MUST RUN setAlias.sh OR setAlias.csh TO RUN ON OLDER KERNELS

If you do not, you will get errors about libraries and binaries not found.

A More in Depth Look
--------------------

After you checkout your copy, you will see two folders: cpu and gpu. As you can 
imagine they represent the cpu and gpu distributions respectively. They are 
practically identical to each other except that the GPU 
version requires the CUDNN_HOME to point where the CUDA DNN libraries resides 
and the wheels are different. Inside each folder, you will next encounter the 
folder py3.x which has the wheels for python 3.4 and python 3.5. They shared
the same install system so we do not need to provide different distros at the
moment.

Inside the folders, you will encounter the following structure:

+ fakeRoot ==> This folder contains file systems required to run on older 
               kernels. If your kernel is newer that 3 then you will not need 
               this folder
+ parallel-netcdf-1.7.0 ==> This folder contains a version of the Parallel 
                            NetCDF file format library. It is used by our data 
                            loaders 
+ py_scripts ==> Contain example scripts to run with our framework
+ user_ops ==> Has the source code for each operator
+ utils ==> Has perl scripts to parse the python version from the environment
+ wheels ==> It contains the tensorflow wheel(s)
- install_mpi_tf.csh and install_mpi_tf.sh ==> The main scripts to install the 
                                               framework
- run_TFEnv.csh and run_TFEnv.sh ==> Scripts to set up the environment if it 
                                     already has been created
- setAlias.csh and setAlias.sh ==> Scripts to set appropiate path and aliases 
                                   for older kernels
- clean_up.sh and clean_up.csh ==> Restore python previous environment if 
                                   needed and destroy the virtual environment

First Test
----------

After you have installed it (and set Alias if appropiated), you should do 
this to test:

```
[py_distro] $ cd py_scripts
[py_distro] $ sbatch -N 1 ./test1.sh
[py_distro] $ sbatch -N 4 ./test4.sh

```

You should start seeing learning rates decreasing and iterations per seconds 
for AlexNet, GoogLeNet, ResNet and InceptionV3.



