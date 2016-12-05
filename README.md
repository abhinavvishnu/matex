MaTEx: Machine Learning Toolkit for Extreme Scale
=================================================

MaTEx is a collection of high performance parallel machine learning and
data mining (MLDM) algorithms, targeted for desktops, supercomputers
and cloud computing systems. 

Getting Started
---------------

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
--conv_layers   | None    | 32 32        | numbers of features in convolutional layers with 5x5 window and stride of 2
--full_layers   | 30      | 200 100 50   | numbers of features in fully connecterd layers
--train_batch   | 128     | 1024         | sample batch size
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

Other Supported Algorithms
--------------------
1) k-means, Spectral Clustering

2) KNN, Support Vector Machines

MaTEx uses Message Passing Interface (MPI), which can be used on
Desktops, Cloud Computing Systems and Supercomputers.

System Software Requirements
-----------------------------
MaTEx bundles required software for parallel computing such as
mpich-3.1. They are automatically built, if they are not found on your system. 

Building MaTEx
--------------
Please refer to the INSTALL file for details.

We have provided a build.sh file in the current directory which will
automate some of the installation of dependencies (the bundled MPI and
GA packages) as well as the installation of MaTEx. 

Publications
------------
1) Fault Tolerant Support Vector Machines. Sameh Shohdy, Abhinav Vishnu, and
Gagan Agrawal. International Conference on Parallel Processing (ICPP'16)

2) Accelerating Deep Learning with Shrinkage and Recall. Shuai Zheng,
Abhinav Vishnu, and Chrish Ding. Arxiv Report.

3) Distributed TensorFlow with MPI. Abhinav Vishnu, Charles Siegel and Jeff
Daily. ArXiv Report.

4) Fault Modeling of Extreme Scale Applications using Machine Learning.
Abhinav Vishnu, Hubertus van Dam, Nathan Tallent, Darren Kerbyson and
Adolfy Hoisie. IEEE International Parallel and Distributed Processing
Symposium (IPDPS), May, 2016 (pdf)

5) Predicting the top and bottom ranks of billboard songs using Machine
Learning. Vivek Datla and Abhinav Vishnu. ArXiv report

6) Fast and Accurate Support Vector Machines on Large Scale Systems.
Abhinav Vishnu, Jeyanthi Narasimhan, and Lawrence Holder, Darren
Kerbyson and Adolfy Hoisie. IEEE Cluster 2015, September, 2015 (pdf).

7) Large Scale Frequent Pattern Mining using MPI One-Sided Model. Abhinav
Vishnu, and Khushbu Agarwal. IEEE Cluster 2015, September, 2015 (pdf).

8) Acclerating k-NN with Hybrid MPI and OpenSHMEM. Jian Lin, Khaled
Hamidouche, Jie Zhang, Xiaoyi Lu, Abhinav Vishnu and Dhabaleswar Panda.
OpenSHMEM Workshop, August, 2015. pdf .

Acknowledgement
---------------

MaTEx is supported by PNNL Analysis in Motion (AIM) initiative and US
Government.


Contributors
------------

Technical Lead: **Abhinav Vishnu**

Current Contributors: Jeff Daily, Charles Siegel, Lindy Rauchenstein,
Junqiao Qiu, Chengcheng Jia

Project Alumni: Sameh Abdulah, Jeyanthi Narasimhan, Joon Hee Choi, Gagan
Agrawala

Support
-------
Email matex-users@googlegroups.com for all questions/bugs/requests.
