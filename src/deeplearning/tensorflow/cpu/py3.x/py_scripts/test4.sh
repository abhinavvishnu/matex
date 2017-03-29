#!/bin/bash
#SBATCH -N 4
#SBATCH -J scaling.test.4gpu
#SBATCH -o scaling.test.4gpu.out.%j
#SBATCH -e scaling.test.4gpu.err.%j

mpirun --map-by node -n 4  --mca opal_event_include poll $FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 --library-path $PNETCDF_INSTALL_DIR/lib:$FAKE_SYSTEM_LIBS/lib/:$FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/:$FAKE_SYSTEM_LIBS/usr/lib64/gconv:$FAKE_SYSTEM_LIBS/usr/lib64/audit:$FAKE_SYSTEM_LIBS/usr/lib64:$LD_LIBRARY_PATH $PYTHONHOME/bin/python $PWD/datasets.py
mpirun --map-by node -n 4  --mca opal_event_include poll $FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 --library-path $PNETCDF_INSTALL_DIR/lib:$FAKE_SYSTEM_LIBS/lib/:$FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/:$FAKE_SYSTEM_LIBS/usr/lib64/gconv:$FAKE_SYSTEM_LIBS/usr/lib64/audit:$FAKE_SYSTEM_LIBS/usr/lib64:$LD_LIBRARY_PATH $PYTHONHOME/bin/python $PWD/lenet3.py --train_batch=16
for network in "AlexNet" "GoogLeNet" "InceptionV3" "ResNet50"
do
    mpirun --map-by node -n 4  --mca opal_event_include poll $FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 --library-path $PNETCDF_INSTALL_DIR/lib:$FAKE_SYSTEM_LIBS/lib/:$FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/:$FAKE_SYSTEM_LIBS/usr/lib64/gconv:$FAKE_SYSTEM_LIBS/usr/lib64/audit:$FAKE_SYSTEM_LIBS/usr/lib64:$LD_LIBRARY_PATH $PYTHONHOME/bin/python $PWD/time_test.py --train_batch=64 --network=$network --iterations=500
done
