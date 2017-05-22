#!/bin/bash
#SBATCH -N 2
#SBATCH -J ImageNet.Hybrid.2
#SBATCH -o ImageNet.Hybrid.2.out.%j
#SBATCH -e ImageNet.Hybrid.2.err.%j

    mpirun --map-by node -n 2  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/hybrid_timetest.py --train_batch 64 --iterations 500 --network "AlexNet"
    mpirun --map-by node -n 2  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/hybrid_timetest.py --train_batch 64 --iterations 500 --network "InceptionV3"
    mpirun --map-by node -n 2  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/hybrid_timetest.py --train_batch 64 --iterations 500 --network "ResNet50"
    mpirun --map-by node -n 2  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/hybrid_timetest.py --train_batch 64 --iterations 500 --network "GoogLeNet"

