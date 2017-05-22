#!/bin/bash
#SBATCH -N 2
#SBATCH -J LeNet.Hybrid.2
#SBATCH -o LeNet.Hybrid.2.out.%j
#SBATCH -e LeNet.Hybrid.2.err.%j

    mpirun --map-by node -n 2  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/hybrid_lenet3.py --train_batch 32 --iterations 1000

