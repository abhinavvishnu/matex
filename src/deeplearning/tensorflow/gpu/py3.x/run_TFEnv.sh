#!/bin/bash

if [ -z ${CUDNN_HOME+x} ]; then
   echo "Need to set CUDNN_HOME to where the CuDNN libraries resides"
   return 2
else
   echo "CUDNN_HOME set to $CUDNN_HOME"
fi


if [ -d $PWD/py_distro ]; then
   source py_distro/bin/activate
   export PYTHONHOME=$PWD/py_distro
fi

PYVRD="$(./utils/strippyd.pl)"

echo -e "\e[32mGuessing Values for the required environment variables\e[0m"
export PNETCDF_INSTALL_DIR=$HOME/opt
export LD_LIBRARY_PATH=$CUDNN_HOME/lib64:$LD_LIBRARY_PATH
export TF_HOME=$PWD/py_distro/lib/python${PYVRD}/site-packages/tensorflow
export TF_INSTALL_DIR=$PWD

echo "Assuming PNETCDF_INSTALL_DIR to be " $PNETCDF_INSTALL_DIR
echo "Assuming TF_HOME to be " $TF_HOME
echo "Assuming TF_INSTALL_DIR to be " $TF_INSTALL_DIR

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
fi

export TF_SCRIPT_HOME=$TF_INSTALL_DIR/py_scripts/
export TF_MPI_ENABLE=1
export FAKE_SYSTEM_LIBS=$TF_INSTALL_DIR/fakeRoot/

