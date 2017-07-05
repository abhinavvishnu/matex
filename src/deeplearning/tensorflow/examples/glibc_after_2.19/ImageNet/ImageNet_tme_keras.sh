#!/bin/bash
#SBATCH -J ImageNet.Keras
#SBATCH -o ImageNet.Keras.out.%j
#SBATCH -e ImageNet.Keras.err.%j

train_batch=128
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

    mpirun --map-by node -n $nodes --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/keras_timetest.py --train_batch $train_batch --iterations 500 --network "AlexNet"
    mpirun --map-by node -n $nodes --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/keras_timetest.py --train_batch $train_batch --iterations 500 --network "InceptionV3"
    mpirun --map-by node -n $nodes --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/keras_timetest.py --train_batch $train_batch --iterations 500 --network "ResNet50"
    mpirun --map-by node -n $nodes --mca opal_event_include poll $PYTHONHOME/bin/python $PWD/keras_timetest.py --train_batch $train_batch --iterations 500 --network "GoogLeNet"

