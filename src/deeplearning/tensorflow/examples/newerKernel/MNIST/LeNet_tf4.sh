#!/bin/bash
#SBATCH -N 4
#SBATCH -J LeNet.TF.4
#SBATCH -o LeNet.TF.4.out.%j
#SBATCH -e LeNet.TF.4.err.%j

    mpirun --map-by node -n 4  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/tf_lenet3.py --train_batch 16 --iterations 1000

