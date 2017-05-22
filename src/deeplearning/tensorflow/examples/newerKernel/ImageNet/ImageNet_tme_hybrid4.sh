#!/bin/bash
#SBATCH -N 4
#SBATCH -J ImageNet.Hybrid.4
#SBATCH -o ImageNet.Hybrid.4.out.%j
#SBATCH -e ImageNet.Hybrid.4.err.%j

    mpirun --map-by node -n 4  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/hybrid_timetest.py --train_batch 32 --iterations 500 --network "AlexNet"
    mpirun --map-by node -n 4  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/hybrid_timetest.py --train_batch 32 --iterations 500 --network "InceptionV3"
    mpirun --map-by node -n 4  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/hybrid_timetest.py --train_batch 32 --iterations 500 --network "ResNet50"
    mpirun --map-by node -n 4  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/hybrid_timetest.py --train_batch 32 --iterations 500 --network "GoogLeNet"

