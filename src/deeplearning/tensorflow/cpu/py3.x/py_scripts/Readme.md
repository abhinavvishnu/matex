MaTEx TensorFlow Scripts
===============================

Included here are Python scripts to assist in building and training deep learning algorithms in TensorFlow.
Additionally, Bash scripts are included to provide tests for these Python scripts and for MaTEx TensorFlow 
itself.

Required Packages
-----------------
* MaTEx TensorFlow, either CPU or GPU version
* Keras 1.2.2 (not compatible with Keras 2.0)
* mpi4py 2.0.0+
* NumPy 1.12.1+
* Six 1.10.0+

Scripts for Data Loading and Network Creation
---

```python
data = DataSet(data_name, 
               train_batch_size=None, 
               test_batch_size=None, 
               normalize=1.0, 
               file1=None, 
               file2=None,
               valid_pct=0.0, 
               test_pct=0.0)
```

datasets.py provides the DataSet class, which reads MNIST, CIFAR, CSV and PNetCDF data in parallel.
The DataSet class takes the following arguments:
* data_name: "MNIST", "CIFAR10", "CIFAR100", "CSV" or "PNETCDF"
* train_batch_size (optional): Size of a training batch for use with next_train_batch method 
* test_batch_size (optional): Size of a testing batch for use with next_validation_batch and next_test_batch methods
* normalize (optional): Float to divide all data entries by (default value is 1.0)
* file1 (optional): Required for CSV or PNETCDF, the location of a file consisting of training data
* file2 (optional): Location of a file consisting of Testing or Validation data for PNETCDF
* valid_pct (optional): Float.  If provided, for non-MNIST or PNETCDF data_name, will place this fraction of loaded data into a validation set
* test_pct (optional): Float.  If provided, for non-MNIST or PNETCDF data_name, will place this fraction of loaded data into a testing set

DataSet class provides the following methods:
```python
[data, labels] = data.next_train_batch()
[data, labels] = data.next_validation_batch()
[data, labels] = data.next_test_batch()
```

None of which take any arguments, each of which returns the next batch (of sizes provided by DataSet class) of both data and labels as a list.

keras_nets.py provides classes for the following networks:
* LeNet3
* AlexNet
* GoogLeNet
* InceptionV3
* ResNet50

e.g.
```python
net = AlexNet()

...

feed_dict={net.x: data, net.y_:labels}

...

loss = categorical_crossentropy(net.y_, net.y)
```

Each of which has properties x, a placeholder for data, y_, a placeholder for labels, and y, the predicted output.

Testing Scripts, Backend Scripts, and Data
---

Run test1.sh and test4.sh on 1 and 4 nodes.

```
sbatch test1.sh
sbatch test4.sh
```

Additional files used for testing:
* pnetcdf.py: helper functions for loading PNetCDF files
* mnist.csv: MNIST Dataset as a CSV for testing
* mnist_train.nc, mnist_test.nc: MNIST Dataset as PNetCDFs for testing
* keras_layers.py: helper Keras layers for constructing networks in keras_nets.py
* test1.sh: testing script for 1 node
* test4.sh: testing script for 4 nodes
* lenet3.py: testing script which trains LeNet3 network
* time_test.py: testing script computing timings for ImageNet model networks
