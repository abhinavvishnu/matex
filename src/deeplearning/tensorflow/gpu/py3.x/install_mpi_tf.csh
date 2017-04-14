#!/bin/tcsh

### Setting up the environment for TensorFlow with MPI
### extensions using C shell. Must be run under the 
### untarred environment.
###
### This script will create if not done already and activate
### a python virtual environment for Tensorflow to run under
### the folder py_distro.
### Depends on python3.4, openmpi/1.8.3 and gcc/4.9.2

### Points to the head of the python virtual environment 
### distribution
set pdistro="$PWD/py_distro"

if(! $?CUDNN_HOME) then
   echo "CUDNN_HOME must be set and point to where the CuDNN library resides"
   exit 2
endif

### Check if the virtual environment has been created
if (-d $pdistro) then
   source $pdistro/bin/activate.csh ### Activate the virt env
### If not create the environment
else
   set py3="`which python3.4`"
   set base="`dirname $py3`"
   set venv="$base/virtualenv"
   set pip="$base/pip"

   if( -f $py3 && -f $venv && -f $pip) then
      echo "Using: $py3, $venv, $pip"
   else
      echo "Failure to find the correct binaries for python, virtualenv or pip"
      exit(1)
   endif

   $venv -p $py3 --always-copy $pdistro
### Make sure that new pip packages are installed in the 
### correct folder
   setenv OLD_PYTHONHOME $PYTHONHOME
   source $pdistro/bin/activate.csh ### Activate the virt env
   setenv PYTHONHOME $pdistro
   set pip = $PYTHONHOME/bin/pip

   $pip install --upgrade pip
### Install the python TF dependencies ensuring that they are
### the latest ones
   $pip install mpi4py numpy scipy --no-cache-dir --upgrade
### OPTIONAL: Install Keras compatible version
   $pip install keras==1.2.2 --no-cache-dir --upgrade
endif
set pip = $PYTHONHOME/bin/pip
### Obtain the version of the current installed python
set PYVRD="`$PWD/utils/strippyd.pl`"

### Guess the locations of necessary environment variables 
### and set them accordingly

echo -e "\e[32mGuessing default values for the required environment variables\e[0m"
### Location of the Parallel NETCDF libraries install
### directories
setenv PNETCDF_INSTALL_DIR $HOME/opt
### The place where the python TensorFlow framework resides
setenv TF_HOME $PWD/py_distro/lib/python${PYVRD}/site-packages/tensorflow
### The current place of wheels and scripts
setenv TF_INSTALL_DIR $PWD
setenv LD_LIBRARY_PATH $CUDNN_HOME/lib64:$LD_LIBRARY_PATH


echo "Assuming PNETCDF_INSTALL_DIR to be " $PNETCDF_INSTALL_DIR
echo "Assuming TF_HOME to be " $TF_HOME
echo "Assuming TF_INSTALL_DIR to be " $TF_INSTALL_DIR

echo -e "\e[93mCheck and update if necessary\e[0m"

### Find the necessary components to install the wheel

## Obtain the python version with a script
set PYVR="`$TF_INSTALL_DIR/utils/strippy.pl`"
### Locate where the TF wheel should be located
set WHEELDIR="$TF_INSTALL_DIR/wheels"
### Name the correct wheel with the correct python version
set WHEEL="$WHEELDIR/tensorflow-1.0.0-cp${PYVR}-cp${PYVR}m-linux_x86_64.whl"

echo "\e[32mInstalling TensorFlow\e[0m"
### Install the tensorflow modified environment
$pip install $WHEEL --upgrade

echo "\e[31mAllocating the user ops dynamic libraries\e[0m"

cd $TF_INSTALL_DIR/user_ops ; make clean ; make; cd $TF_INSTALL_DIR

### Make sure that the shared objects are available

if(-f $TF_INSTALL_DIR/user_ops/tf_reduce.so) then
   echo "\e[32mTF_REDUCE: Dynamic Library found\e[0m"
else
   echo "\e[31mTF_REDUCE: Dynamic Library not found\e[0m"
   exit (1)
endif

if(-f $TF_INSTALL_DIR/user_ops/tf_broadcast.so) then
   echo "\e[32mTF_BROADCAST: Dynamic Library found\e[0m"
else
   echo "\e[31mTF_BROADCAST: Dynamic Library not found\e[0m"
   exit (1)
endif

## Copy the shared objects to the correct locations

cp -r $TF_INSTALL_DIR/user_ops $TF_HOME/core/

### Optional: Compile and install the PNETCDF library to be
### used by the parallel readers

echo "\e[32mCompiling PNETCDF\e[0m"

setenv MPICC "`which mpicc`"

cd $TF_INSTALL_DIR/parallel-netcdf-1.7.0
./configure --prefix=$PNETCDF_INSTALL_DIR CFLAGS="-g -O2 -fPIC" CPPFLAGS="-g -O2 -fPIC" CXXFLAGS="-g -O2 -fPIC" FFLAGS="-O2 -fPIC" FCFLAGS="-O2 -fPIC" --disable-cxx --disable-fortran
make clean ; make
make install 
make shared_library
cp ./src/lib/libpnetcdf.so $PNETCDF_INSTALL_DIR/lib/
cd $TF_INSTALL_DIR

### Because the way that the user ops work, we might need to 
### preload the MPI library, set the correct library to the
### LD_PRELOAD variable
echo "\e[32mSetting dynamic load MPI library\e[0m"

set dir = "`which mpicc`"
set dirn = "`dirname $dir`"
set name1 = "/../lib/libmpi_cxx.so"
set name2 = "/../lib64/libmpi_cxx.so"

set full1 = $dirn$name1
set full2 = $dirn$name2

if(-f $full1) then
   setenv LD_PRELOAD "$full1"
   echo "\e[32mPreload library is set to $LD_PRELOAD\e[0m"
else if (-f $full2) then
   setenv LD_PRELOAD "$full2"
   echo "\e[32mPreload library is set to $LD_PRELOAD\e[0m"
else
   echo "\e[31mCannot find the MPI CXX library. Need it for the extension to work correctly\e[0m"
   exit (1)
endif

setenv TF_SCRIPT_HOME $TF_INSTALL_DIR/py_scripts/
setenv TF_MPI_ENABLE 1 

echo "\e[32mDone\e[0m"

