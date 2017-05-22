#!/bin/bash
#SBATCH -N 8
#SBATCH -J ImageNet.Keras.8
#SBATCH -o ImageNet.Keras.8.out.%j
#SBATCH -e ImageNet.Keras.8.err.%j

    mpirun --map-by node -n 8  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/keras_timetest.py --train_batch 16 --iterations 500 --network "AlexNet"
    mpirun --map-by node -n 8  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/keras_timetest.py --train_batch 16 --iterations 500 --network "InceptionV3"
    mpirun --map-by node -n 8  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/keras_timetest.py --train_batch 16 --iterations 500 --network "ResNet50"
    mpirun --map-by node -n 8  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/keras_timetest.py --train_batch 16 --iterations 500 --network "GoogLeNet"

