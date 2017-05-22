#!/bin/bash
#SBATCH -N 4
#SBATCH -J LeNet.Hybrid.4
#SBATCH -o LeNet.Hybrid.4.out.%j
#SBATCH -e LeNet.Hybrid.4.err.%j

    mpirun --map-by node -n 4  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/hybrid_lenet3.py --train_batch 16 --iterations 1000

