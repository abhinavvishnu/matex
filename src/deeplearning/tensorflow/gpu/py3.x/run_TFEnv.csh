#!/bin/tcsh

set pdistro="$PWD/py_distro/"

if(! $?CUDNN_HOME) then
   echo "CUDNN_HOME must be set and point to where the CuDNN library resides"
   exit 2
endif

if (-d $pdistro) then
    source $pdistro/bin/activate.csh
    setenv PYTHONHOME=$PWD/py_distro
fi

set PYVRD="`./utils/strippyd.pl`"

echo "\e[32mGuessing Values for the required environment variables\e[0m"
setenv PNETCDF_INSTAL_DIR $HOME/opt
setenv LD_LIBRARY_PATH $CUDNN_HOME/lib64:$LD_LIBRARY_PATH
setenv TF_HOME $PWD/py_distro/lib/python${PYVRD}/site-packages/tensorflow
setenv TF_INSTALL_DIR $PWD


echo "Assuming PNETCDF_INSTAL_DIR to be " $PNETCDF_INSTAL_DIR
echo "Assuming TF_HOME to be " $TF_HOME
echo "Assuming TF_INSTALL_DIR to be " $TF_INSTALL_DIR

echo "\e[32mSetting dynamic load MPI library\e[0m"

set dir = "`which mpicc`"
set dirn = "`dirname $dir`"
set name1 = "/../lib/libmpi_cxx.so"
set name2 = "/../lib64/libmpi_cxx.so"

set full1 = $dirn$name1
set full2 = $dirn$name2

if(-f $full1)
   setenv LD_PRELOAD "$full1"
   echo "\e[32mPreload library is set to $LD_PRELOAD\e[0m"
elif (-f $full2)
   setenv LD_PRELOAD "$full2"
   echo "\e[32mPreload library is set to $LD_PRELOAD\e[0m"
else
   echo "\e[31mCannot find the MPI CXX library. Need it for the extension to work correctly\e[0m"
endif

setenv TF_SCRIPT_HOME $TF_INSTALL_DIR/py_scripts/
setenv TF_MPI_ENABLE 1
setenv FAKE_SYSTEM_LIBS $TF_INSTALL_DIR/fakeRoot/
