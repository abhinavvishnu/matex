#!/bin/bash
#SBATCH -N 2
#SBATCH -J LeNet.Keras.2
#SBATCH -o LeNet.Keras.2.out.%j
#SBATCH -e LeNet.Keras.2.err.%j

    mpirun --map-by node -n 2  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/keras_lenet3.py --train_batch 32

