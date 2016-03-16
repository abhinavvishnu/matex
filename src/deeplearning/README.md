Deep Learning using TensorFlow and MPI (mpi-tensorflow.py)
==========================================================

mpi-tensorflow.py uses the Message Passing Interface (MPI) via mpi4py to replicate a TensorFlow graph.  Each MPI rank receives a disjoint partition of the input dataset, processes the local samples, and performs an MPI reduction of the weights and biases before entering the next epoch.

Datasets
--------

mpi-tensorflow.py will default to using the MNIST dataset.  Other datasets can be selected using the --data command-line parameter (see below).  The first time the MNIST, CIFAR10, or CIFAR100 datasets are used, they will be downloaded.  Subsequent runs will reuse the downloaded data and only extract it.

User-specified data is given by the `--data CSV` and `--filename <file.csv>` command-line parameters.  It is expected that the CSV file contains the classifier as the first column.  The validation and testing sets are created by reserving a percentage of the samples from the end of the data set using `--valid_pct` and `--test_pct`, respectively.

Command-Line Interface
----------------------

Parameter       | Default | Alt. Example | Description
----------------| ------- | ------------ | -----------
--conv_layers   | None    | 32,32        | convolution layers
--full_layers   | 30      | 200,100,50   | fully connected layers
--train_batch   | 10      | 1000         | sample batch size
--epochs        | 30      | 300          | number of epochs to run
--time          | -1      | 60.0         | stop running after given number of seconds; when present, --epochs is ignored
--learning_rate | 0.01    | 1.1          | learning rate
--data          | MNIST   | CIFAR10      | data set, one of MNIST, CIFAR10, CIFAR100, CSV
--inter_threads | 1       | 5            | sets inter_threads used by TensorFlow
--intra_threads | 1       | 5            | sets intra_threads used by TensorFlow
--threads       | 1       | 5            | sets both inter_threads and intra_threads
--filename      | None    | file.csv     | filename to use with --data CSV
--valid_pct     | 0.1     | 0.05         | percentage of samples to reserve for validation set (used by all data sets)
--test_pct      | 0.1     | 0.05         | percentage of samples to reserve for testing set (used only by --data CSV)
