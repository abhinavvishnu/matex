#!/bin/bash
#SBATCH -N 1
#SBATCH -J ImageNet.Keras.1
#SBATCH -o ImageNet.Keras.1.out.%j
#SBATCH -e ImageNet.Keras.1.err.%j

    mpirun --map-by node -n 1  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/keras_timetest.py --train_batch 128 --iterations 500 --network "AlexNet"
    mpirun --map-by node -n 1  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/keras_timetest.py --train_batch 128 --iterations 500 --network "InceptionV3"
    mpirun --map-by node -n 1  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/keras_timetest.py --train_batch 128 --iterations 500 --network "ResNet50"
    mpirun --map-by node -n 1  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/keras_timetest.py --train_batch 128 --iterations 500 --network "GoogLeNet"

