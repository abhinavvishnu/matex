#!/bin/bash
#SBATCH -N 1
#SBATCH -J LeNet.TF.1
#SBATCH -o LeNet.TF.1.out.%j
#SBATCH -e LeNet.TF.1.err.%j

    mpirun --map-by node -n 1  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/tf_lenet3.py --train_batch 64 --iterations 1000

