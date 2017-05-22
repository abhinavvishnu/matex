#!/bin/bash
#SBATCH -N 2
#SBATCH -J LeNet.TF.2
#SBATCH -o LeNet.TF.2.out.%j
#SBATCH -e LeNet.TF.2.err.%j

    mpirun --map-by node -n 2  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/tf_lenet3.py --train_batch 32 --iterations 1000

