#!/bin/bash

### Setting up the environment for TensorFlow with MPI
### extensions using bash shell. Must be run under the
### untarred environment.
###
### This script will create if not done already and activate
### a python virtual environment for Tensorflow to run under
### the folder py_distro.
### Depends on python3.5, openmpi/1.8.3 and above and gcc/4.8.5 and above
if [[ -z "${ANACONDA_HOME}" ]]; then
echo "ANACONDA_HOME environment variable not set"
return 1
else
export PATH=${ANACONDA_HOME}/bin:$PATH
fi

if [[ -z "${MPI_HOME}" ]]; then
echo "MPI_HOME environment variable not set"
return 1
fi

if [[ -z "${MTX_INSTALL_DIR}" ]]; then
  INSTDIR=$PWD
else
  INSTDIR=${MTX_INSTALL_DIR}
fi

if [ -d $INSTDIR/matex_tf ]; then
   source activate $INSTDIR/matex_tf
else
   py3=$(which python3)
   base=$(dirname $py3)

   pip="$base/pip"

   if [ -f $py3 ] && [ -f $pip ]; then
      echo -e "Using: $py3, $pip"
   else
      echo -e "Failure to find the correct binaries for python or pip"
      return 1
   fi
   export OLD_PYTHONHOME=$PYTHONHOME
   conda create -p ${INSTDIR}/matex_tf python=3.5.2 conda anaconda -y
   source activate ${INSTDIR}/matex_tf

   export PYTHONHOME=${INSTDIR}/matex_tf
   pip="$PYTHONHOME/bin/pip"
   $pip  install pip --upgrade
   $pip install mpi4py numpy scipy --upgrade --no-cache-dir
fi

PYVRD="$(python $PWD/utils/strippyd.py)"

echo -e "\e[32mGuessing Values for the required environment variables\e[0m"

export PNETCDF_INSTALL_DIR=${INSTDIR}/opt
export TF_HOME=${INSTDIR}/matex_tf/lib/python${PYVRD}/site-packages/tensorflow
export TF_INSTALL_DIR=$PWD

echo "Assuming PNETCDF_INSTALL_DIR to be " $PNETCDF_INSTALL_DIR
echo "Assuming TF_HOME to be " $TF_HOME
echo "Assuming TF_INSTALL_DIR to be " $TF_INSTALL_DIR

echo -e "\e[93mCheck and update if necessary\e[0m"

PYVR="$(python $TF_INSTALL_DIR/utils/strippy.py)"
PYVRD="$(python $TF_INSTALL_DIR/utils/strippyd.py)"
WHEELDIR="$TF_INSTALL_DIR/wheels/"
WHEEL="$WHEELDIR/tensorflow-1.7.0-cp${PYVR}-cp${PYVR}m-linux_x86_64.whl"

echo -e "\e[32mInstalling MPI Tensorflow\e[0m"

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

if [ -f $TF_INSTALL_DIR/user_ops/tf_sync.so ]; then
   echo -e "\e[32mSync operations built\e[0m"
else
   echo -e "\e[32mSync operation failed to build\e[0m"
   return 1
fi

cp -r $TF_INSTALL_DIR/user_ops $TF_HOME/core/

cp -r $TF_INSTALL_DIR/utils/matex $TF_HOME/python/
if [ -d  $TF_HOME/python/matex ]; then
   echo "Successfully installed the matex headers"
else
   echo "Failed to copy the correct folders to $TF_HOME/python/matex"
   return 12
fi

echo -e "\e[32mCompiling PNETCDF\e[0m"

cd ${TF_INSTALL_DIR}/parallel-netcdf-1.7.0
mkdir -p ./build
cd ./build
export MPICC=$(which mpicc)
../configure --prefix=$PNETCDF_INSTALL_DIR CFLAGS="-g -O2 -fPIC" CPPFLAGS="-g -O2 -fPIC" CXXFLAGS="-g -O2 -fPIC" FFLAGS="-O2 -fPIC" FCFLAGS="-O2 -fPIC" --disable-cxx --disable-fortran > /dev/null 2>&1
make clean > /dev/null 2>&1 ; make > /dev/null 2>&1
make install > /dev/null 2>&1
make shared_library > /dev/null 2>&1
cp ./src/lib/libpnetcdf.so $PNETCDF_INSTALL_DIR/lib/
cd $TF_INSTALL_DIR

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
