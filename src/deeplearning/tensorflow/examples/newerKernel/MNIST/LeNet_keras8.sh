#!/bin/bash
#SBATCH -N 8
#SBATCH -J LeNet.Keras.8
#SBATCH -o LeNet.Keras.8.out.%j
#SBATCH -e LeNet.Keras.8.err.%j

    mpirun --map-by node -n 8  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/keras_lenet3.py --train_batch 8 --iterations 1000

