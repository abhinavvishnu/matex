#!/bin/bash
#SBATCH -N 8
#SBATCH -J LeNet.Hybrid.8
#SBATCH -o LeNet.Hybrid.8.out.%j
#SBATCH -e LeNet.Hybrid.8.err.%j

    mpirun --map-by node -n 8  --mca opal_event_include poll $FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 --library-path $PNETCDF_INSTALL_DIR/lib:$FAKE_SYSTEM_LIBS/lib/:$FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/:$FAKE_SYSTEM_LIBS/usr/lib64/gconv:$FAKE_SYSTEM_LIBS/usr/lib64/audit:$LD_LIBRARY_PATH $PYTHONHOME/bin/python $PWD/hybrid_lenet3.py --train_batch 8 --iterations 1000

