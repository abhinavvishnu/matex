#!/bin/bash
#SBATCH -N 1
#SBATCH -J LeNet.Hybrid.1
#SBATCH -o LeNet.Hybrid.1.out.%j
#SBATCH -e LeNet.Hybrid.1.err.%j

    mpirun --map-by node -n 1  --mca opal_event_include poll $FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 --library-path $PNETCDF_INSTALL_DIR/lib:$FAKE_SYSTEM_LIBS/lib/:$FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/:$FAKE_SYSTEM_LIBS/usr/lib64/gconv:$FAKE_SYSTEM_LIBS/usr/lib64/audit:$LD_LIBRARY_PATH $PYTHONHOME/bin/python $PWD/hybrid_lenet3.py --train_batch 64 --iterations 1000

