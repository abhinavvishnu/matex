#!/bin/bash
#SBATCH -N 1
#SBATCH -J LeNet.Hybrid.1
#SBATCH -o LeNet.Hybrid.1.out.%j
#SBATCH -e LeNet.Hybrid.1.err.%j

    mpirun --map-by node -n 1  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/hybrid_lenet3.py --train_batch 64 --iterations 1000

