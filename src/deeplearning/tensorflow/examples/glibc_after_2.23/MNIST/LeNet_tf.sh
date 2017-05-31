#!/bin/bash
#SBATCH -N 1
#SBATCH -J LeNet.TF.1
#SBATCH -o LeNet.TF.1.out.%j
#SBATCH -e LeNet.TF.1.err.%j

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
echo "Using $nodes for computing with a batch size

    mpirun --map-by node -n $(nodes)  --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/tf_lenet3.py --train_batch $(train_batch) --iterations 1000

