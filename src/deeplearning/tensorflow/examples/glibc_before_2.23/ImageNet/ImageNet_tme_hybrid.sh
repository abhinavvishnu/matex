#!/bin/bash
#SBATCH -N 1
#SBATCH -J ImageNet.Hybrid
#SBATCH -o ImageNet.Hybrid.out.%j
#SBATCH -e ImageNet.Hybrid.err.%j

train_batch=128
nodes=1

while getopts ":b:n:" opt; do
   case $opt in 
      b)
         train_batch=$OPTARG
         ;;
      n)
         nodes=$OPTARG
         ;;
      :)
         echo "Option -$OPTARG requires an Argument"
         exit 1
         ;;
      \?)
         echo "Unknown option: $OPTARG"
         exit 1
         ;;
   esac
done

train_batch=`expr $train_batch / $nodes`
echo "Using $nodes for computing with a batch size of $train_batch"
exit

    mpirun --map-by node -n $(nodes)  --mca opal_event_include poll $FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 --library-path $PNETCDF_INSTALL_DIR/lib:$FAKE_SYSTEM_LIBS/lib/:$FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/:$FAKE_SYSTEM_LIBS/usr/lib64/gconv:$FAKE_SYSTEM_LIBS/usr/lib64/audit:$LD_LIBRARY_PATH $PYTHONHOME/bin/python $PWD/hybrid_timetest.py --train_batch $(train_batch) --iterations 500 --network "AlexNet"
    mpirun --map-by node -n $(nodes)  --mca opal_event_include poll $FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 --library-path $PNETCDF_INSTALL_DIR/lib:$FAKE_SYSTEM_LIBS/lib/:$FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/:$FAKE_SYSTEM_LIBS/usr/lib64/gconv:$FAKE_SYSTEM_LIBS/usr/lib64/audit:$LD_LIBRARY_PATH $PYTHONHOME/bin/python $PWD/hybrid_timetest.py --train_batch $(train_batch) --iterations 500 --network "InceptionV3"
    mpirun --map-by node -n $(nodes)  --mca opal_event_include poll $FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 --library-path $PNETCDF_INSTALL_DIR/lib:$FAKE_SYSTEM_LIBS/lib/:$FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/:$FAKE_SYSTEM_LIBS/usr/lib64/gconv:$FAKE_SYSTEM_LIBS/usr/lib64/audit:$LD_LIBRARY_PATH $PYTHONHOME/bin/python $PWD/hybrid_timetest.py --train_batch $(train_batch) --iterations 500 --network "ResNet50"
    mpirun --map-by node -n $(nodes)  --mca opal_event_include poll $FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 --library-path $PNETCDF_INSTALL_DIR/lib:$FAKE_SYSTEM_LIBS/lib/:$FAKE_SYSTEM_LIBS/lib/x86_64-linux-gnu/:$FAKE_SYSTEM_LIBS/usr/lib64/gconv:$FAKE_SYSTEM_LIBS/usr/lib64/audit:$LD_LIBRARY_PATH $PYTHONHOME/bin/python $PWD/hybrid_timetest.py --train_batch $(train_batch) --iterations 500 --network "GoogLeNet"

