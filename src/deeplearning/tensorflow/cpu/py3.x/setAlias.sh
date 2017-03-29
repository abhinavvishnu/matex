#!/usr/bin/bash

### Set environment to run TensorFlow in systems with older
### kernels

if [ -z $PNETCDF_INSTALL_DIR ]; then
   echo "Need to set the Parallel NetCDF directory"
   exit 1
fi

if [ -z $TF_HOME ]; then
   echo "Need to set the location of the tensorflow folder (probably the directory where tensorflow was installed under python)"
   exit 1
fi

if [ -z $TF_INSTALL_DIR ]; then
   echo "Need to set the location of the source folder (probably the directory where this script resides)"
   exit 1
fi

## Point to the fake libraries folders. This folder has support
## for loader 2.23 

export FAKE_SYSTEM_LIBS=$TF_INSTALL_DIR/fakeRoot/

### Set alias to run tensorflow interactively. If run inside
### scripts, plese note that python must be run as shown below
### See py_scripts for examples

alias pyflow="$FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 --library-path $PNETCDF_INSTALL_DIR/lib:$FAKE_SYSTEM_LIBS/lib/:$FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/:$FAKE_SYSTEM_LIBS/usr/lib64/gconv:$FAKE_SYSTEM_LIBS/usr/lib64/audit:$FAKE_SYSTEM_LIBS/usr/lib64:$LD_LIBRARY_PATH $TF_INSTALL_DIR/py_distro/bin/python3.4"

