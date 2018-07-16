#!/bin/bash

### Setting up the environment for TensorFlow with MPI
### extensions using bash shell. Must be run under the
### untarred environment.
###
### This script will create if not done already and activate
### a python virtual environment for Tensorflow to run under
### the folder py_distro.
### Depends on python3.4, openmpi/1.8.3 and gcc/4.9.2

if [ -z ${CUDNN_HOME+x} ]; then
   echo "Need to set CUDNN_HOME to where the CuDNN libraries resides"
   return 2
else
   echo "CUDNN_HOME set to $CUDNN_HOME"
fi

if [ -d $PWD/py_distro ]; then
   source $PWD/py_distro/bin/activate
else
   py3=$(which python3.4)
   venv=$(which virtualenv)
   base=$(dirname $py3)
   set pip="$base/pip"
   if [ -f $py3 ] && [ -f $venv ]; then
      echo "Using: $py3, $venv, $pip"
   else
      echo "Failure to find the correct binaries for python, virtualenv or pip, $py3 - $pip - $venv"
      return 1
   fi
   export OLD_PYTHONHOME=$PYTHONHOME
   $venv -p $(which python3.4) py_distro
   source py_distro/bin/activate
   export PYTHONHOME=$PWD/py_distro
   pip="$PYTHONHOME/bin/pip"
   $pip install pip --upgrade
   $pip install mpi4py numpy scipy --upgrade --no-cache-dir
   $pip install keras==1.2.2 --no-cache-dir --upgrade
fi

PYVRD="$(python $PWD/utils/strippyd.py)"

echo -e "\e[32mGuessing Values for the required environment variables\e[0m"

keras_backend="$PYTHONHOME/lib/python3.4/site-packages/keras/backend/tensorflow_backend.py"

## Patch the keras distribution with the tensorflow enhanced version of the backend

if [ -f $keras_backend ]; then
   cp $PWD/utils/tensorflow_backend.py $keras_backend
else
   echo "Incomplete or failed keras installation"
   return 11
fi

export PNETCDF_INSTALL_DIR=$HOME/opt
export TF_HOME=$PWD/py_distro/lib/python${PYVRD}/site-packages/tensorflow
export TF_INSTALL_DIR=$PWD
export LD_LIBRARY_PATH=$CUDNN_HOME/lib64:$LD_LIBRARY_PATH

echo "Assuming PNETCDF_INSTALL_DIR to be " $PNETCDF_INSTALL_DIR
echo "Assuming TF_HOME to be " $TF_HOME
echo "Assuming TF_INSTALL_DIR to be " $TF_INSTALL_DIR

echo -e "\e[93mCheck and update if necessary\e[0m"

PYVR="$(python $TF_INSTALL_DIR/utils/strippy.py)"
PYVRD="$(python $TF_INSTALL_DIR/utils/strippyd.py)"
WHEELDIR="$TF_INSTALL_DIR/wheels/"
TF_VERSION="1.0.0"

WHEELDIR="$WHEELDIR/8.0/"

WHEEL="$WHEELDIR/tensorflow_gpu-1.0.0-cp${PYVR}-cp${PYVR}m-linux_x86_64.whl"

echo -e "\e[32mInstalling MPI Tensorflow"

if [ -f $WHEEL ]; then
   echo -e "\e[32mWheel found Successfully\e[0m"
else
   echo -e "\e[93mWheel was not found\e[0m"
   return 1
fi

$pip install $WHEEL --upgrade

echo -e "\e[32mInstalling User Ops\e[0m"

cd $TF_INSTALL_DIR/user_ops; make clean ; make ; cd $TF_INSTALL_DIR

if [ -f $TF_INSTALL_DIR/user_ops/tf_reduce.so ]; then
   echo -e "\e[32mReduce operations built\e[0m"
else
   echo -e "\e[93mReduce operation failed to build\e[0m"
   return 1
fi

if [ -f $TF_INSTALL_DIR/user_ops/tf_broadcast.so ]; then
   echo -e "\e[32mBroadcast operations built\e[0m"
else
   echo -e "\e[32mBroadcast operation failed to build\e[0m"
   return 1
fi

if [ -f $TF_INSTALL_DIR/user_ops/tf_bind.so ]; then
   echo -e "\e[32mBind operations built\e[0m"
else
   echo -e "\e[32mBind operation failed to build\e[0m"
   return 1
fi

cp -r $TF_INSTALL_DIR/user_ops $TF_HOME/core/

echo -e "\e[32mCompiling PNETCDF\e[0m"

cd ./parallel-netcdf-1.7.0
export MPICC=$(which mpicc)
./configure --prefix=$PNETCDF_INSTALL_DIR CFLAGS="-g -O2 -fPIC" CPPFLAGS="-g -O2 -fPIC" CXXFLAGS="-g -O2 -fPIC" FFLAGS="-O2 -fPIC" FCFLAGS="-O2 -fPIC" --disable-cxx --disable-fortran > /dev/null 2>&1
make clean > /dev/null 2>&1 ; make > /dev/null 2>&1
make install > /dev/null 2>&1
make shared_library > /dev/null 2>&1
cp ./src/lib/libpnetcdf.so $PNETCDF_INSTALL_DIR/lib/
cd ..

if [ -f  $PNETCDF_INSTALL_DIR/lib/libpnetcdf.so ]; then
   echo -e "\e[32mSuccessfully installed the PNETCDF library\e[0m"
else
   echo -e "\e[93mFailed to install the PNETCDF library\e[0m"
   return 12
fi

echo -e "\e[32mSetting dynamic load MPI library\e[0m"

ndir=$(which mpicc)
nndir=$(dirname $ndir)
name1="/../lib/libmpi_cxx.so"
name2="/../lib64/libmpi_cxx.so"
full1=$nndir$name1
full2=$nndir$name2

if [ -f $full1 ]; then
   export LD_PRELOAD="$full1"
elif [ -f $full2 ]; then
   export LD_PRELOAD="$full2"
else
   echo -e "\e[32mCannot find the MPI CXX library. Need it for the extensions to work correctly\e[0m"
   return 0
fi

export TF_SCRIPT_HOME=$TF_INSTALL_DIR/../../examples
export TF_MPI_ENABLE=1

echo "Done ..."
