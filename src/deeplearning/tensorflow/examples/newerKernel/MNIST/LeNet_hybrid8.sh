#!/bin/bash
#SBATCH -N 8
#SBATCH -J LeNet.Hybrid.8
#SBATCH -o LeNet.Hybrid.8.out.%j
#SBATCH -e LeNet.Hybrid.8.err.%j

    mpirun --map-by node -n 8  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/hybrid_lenet3.py --train_batch 8 --iterations 1000

