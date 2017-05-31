#!/bin/bash
#SBATCH -J LeNet.Hybrid
#SBATCH -o LeNet.Hybrid.out.%j
#SBATCH -e LeNet.Hybrid.err.%j

train_batch=64
nodes=1

while getopts ":b:n:" opt; do
   case $opt in
      b)
         train_batch=$OPTARG
         ;;
      n)
         nodes=$OPTARG
         ;;
      :)
         echo "Option -$OPTARG requires an Argument"
         exit 1
         ;;
      \?)
         echo "Unknown option: $OPTARG"
         exit 1
         ;;
   esac
done

train_batch=`expr $train_batch / $nodes`
echo "Using $nodes for computing with a batch size $train_batch"

    mpirun --map-by node -n $nodes  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/hybrid_lenet3.py --train_batch $train_batch --iterations 1000

