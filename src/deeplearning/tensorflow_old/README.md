Getting Started
==========================================================

mpi-tensorflow.py scales TensorFlow 0.12 on large scale systems using Message Passing Interface (MPI) via mpi4py.  Each MPI rank receives a disjoint partition of the input dataset, processes the local samples, and performs an MPI reduction of the gradients for each iteration.

Example Commands
----------------

The following commands will train LeNet-3 and a convolution network for CIFAR10 respectively.

mpirun -n 2 python mpi-tensorflow.py --data MNIST --conv_layers 20 50 --full_layers 500 --epochs 13 --train_batch 32

mpirun -n 2 python mpi-tensorflow.py --data CIFAR10 --conv_layers 32 32 64 --full_layers 64 --epochs 12 --train_batch 32

Dataset Formats
---------------

MaTEx TensorFlow supports MNIST and CIFAR binaries, CSV and PNetCDF formats.  By default, it will download and train with MNIST.

The first time the MNIST, CIFAR10, or CIFAR100 datasets are used, they will be downloaded.  Subsequent runs will reuse the downloaded data and only extract it.

User-specified data is given by the `--data CSV` and `--data PNETCDF` command-line parameters.  CSV also requires `--filename <file.csv>` where <file.csv> is a CSV file with the class label in the first column.  It will be partitioned into a training, validation and testing set using `--valid_pct` and `--test_pct`, respectively.  PNetCDF requires `--filename <file1.nc>` and `--filename2 <file2.nc>` where <file1.nc> and <file2.nc> are NetCDF files corresponding to training and testing data, respectively.

Command-Line Interface
----------------------

The following table lists command-line parameters, the default value, an additional example value, and a description of what they do.

Parameter       | Default | Alt. Example | Description
----------------| ------- | ------------ | -----------
--conv_layers   | None    | 32 32        | convolution layers
--full_layers   | 30      | 200 100 50   | fully connected layers
--train_batch   | 10      | 1024         | sample batch size
--epochs        | 30      | 300          | number of epochs to run
--time          | -1      | 60.0         | stop running after given number of seconds; when present, --epochs is ignored
--learning_rate | 0.1     | 1.1          | learning rate
--data          | MNIST   | CIFAR10      | data set, one of MNIST, CIFAR10, CIFAR100, CSV or PNETCDF
--inter_threads | 0       | 5            | sets inter_threads used by TensorFlow, when 0 is set dynamically by TensorFlow
--intra_threads | 0       | 5            | sets intra_threads used by TensorFlow, when 0 is set dynamically by TensorFlow
--threads       | 0       | 5            | sets both inter_threads and intra_threads, when 0 is set dynamically by TensorFlow
--filename      | None    | file.csv     | filename to use with --data CSV and --data PNETCDF
--filename      | None    | file.nc      | filename to use with --data PNETCDF
--valid_pct     | 0.1     | 0.05         | fraction of samples to reserve for validation set (used by all data sets)
--test_pct      | 0.1     | 0.05         | fraction of samples to reserve for testing set (used only by --data CSV)
--error_batch   | False   | True         | if set, breaks error and testing into batches.  Does not take an argument.
--top           | 1       | 5            | sets how many of the top rated outputs will be checked against the label for correctness

--------

mpi-tensorflow_alexnet.py is a script that will train AlexNet from scratch.  It only takes two parameters --train_data and --test_data with the arguments a pair of PNetCDF files containing the ILSVRC2012 training and validation sets.
