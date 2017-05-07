#!/bin/bash

### Setting up the environment for TensorFlow with MPI
### extensions using bash shell. Must be run under the 
### untarred environment.
###
### This script will create if not done already and activate
### a python virtual environment for Tensorflow to run under
### the folder py_distro.
### Depends on python3.4, openmpi/1.8.3 and gcc/4.9.2

if [ -d $PWD/py_distro ]; then
   source $PWD/py_distro/bin/activate
else
   py3=$(which python3.4)
   base=$(dirname $py3)
   venv="$base/virtualenv"
   if [ -f $py3 ] && [ -f $venv ]; then
      echo "Using: $py3, $venv, $pip"
   else
      echo "Failure to find the correct binaries for python, virtualenv or pip"
      return 1
   fi
   export OLD_PYTHONHOME=$PYTHONHOME
   $venv -p $(which python3.4) --always-copy py_distro
   source $PWD/py_distro/bin/activate
   export PYTHONHOME=$PWD/py_distro
   pip="$PYTHONHOME/bin/pip"
   $pip install mpi4py numpy scipy --upgrade --no-cache-dir
   $pip install keras==1.2.2 --no-cache-dir --upgrade
fi

PYVRD="$($PWD/utils/strippyd.pl)"

echo -e "\e[32mGuessing Values for the required environment variables\e[0m"

export PNETCDF_INSTALL_DIR=$HOME/opt
export TF_HOME=$PWD/py_distro/lib/python${PYVRD}/site-packages/tensorflow
export TF_INSTALL_DIR=$PWD

echo "Assuming PNETCDF_INSTALL_DIR to be " $PNETCDF_INSTALL_DIR
echo "Assuming TF_HOME to be " $TF_HOME
echo "Assuming TF_INSTALL_DIR to be " $TF_INSTALL_DIR

echo -e "\e[93mCheck and update if necessary\e[0m"

PYVR="$($TF_INSTALL_DIR/utils/strippy.pl)"
PYVRD="$($TF_INSTALL_DIR/utils/strippyd.pl)"
WHEELDIR="$TF_INSTALL_DIR/wheels/"
WHEEL="$WHEELDIR/tensorflow-1.0.0-cp${PYVR}-cp${PYVR}m-linux_x86_64.whl"

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
    echo "\e[93mReduce operation failed to build\e[0m"
    return 1
fi

if [ -f $TF_INSTALL_DIR/user_ops/tf_broadcast.so ]; then  
   echo "\e[32mBroadcast operations built\e[0m"
else
   echo "\e[32mBroadcast operation failed to build\e[0m"
   return 1
fi

if [ -f $TF_INSTALL_DIR/user_ops/tf_sync.so ]; then 
   echo "\e[32mSync operations built\e[0m"
else
   echo "\e[32mSync operation failed to build\e[0m"
   return 1
fi

cp -r $TF_INSTALL_DIR/user_ops $TF_HOME/core/

echo -e "\e[32mCompiling PNETCDF\e[0m"

cd ./parallel-netcdf-1.7.0
export MPICC=$(which mpicc)
./configure --prefix=$PNETCDF_INSTALL_DIR CFLAGS="-g -O2 -fPIC" CPPFLAGS="-g -O2 -fPIC" CXXFLAGS="-g -O2 -fPIC" FFLAGS="-O2 -fPIC" FCFLAGS="-O2 -fPIC" --disable-cxx --disable-fortran
make clean ; make
make install
make shared_library
cp ./src/lib/libpnetcdf.so $PNETCDF_INSTALL_DIR/lib/
cd ..


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


