#!/bin/bash

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

export TF_MPI_ENABLE=1
export LD_PRELOAD=${MPI_HOME}/lib/libmpi.so:$MPI_HOME/lib/libmpi_cxx.so
source activate $PWD/matex_tf
export PYTHONHOME=$PWD/matex_tf
