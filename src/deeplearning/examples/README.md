TensorFlow 0.7.1 vs TensorFlow 0.8.0 vs Caffe
==========================================================

We have compared the performance of TensorFlow and Caffe for convolutions on CPUs.

Software         | Average Time per Epoch
-----------------| ----------------------
TensorFlow 0.7.1 | 43.066143663s
TensorFlow 0.8.0 | 63.3018684705s
Caffe            | 20.8077090979s

The tested network used the full MNIST training set of 60000 greyscale 28 x 28 images and contained a single convolution layer with 5x5 window and 20 features, followed by a 2x2 max pooling layer with stride 2 and a ReLU nonlinearity.  The output from this layer was fully-connected to a softmax output layer and optimized with cross-entropy as its loss function, with training batch size 10 and testing batch size 100.

These tests were performed on a Windows 7 enterprise sp 1 (64-bit) computer with an Ubuntu 15.10LTS virtual box with access to 5292 MB of RAM and 4 processors from an Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz with 8.00 GB of RAM.

The TensorFlow 0.8.0 and PyCaffe files are included as tensorflow_test.py and pycaffe_test.py.  The only difference between TensorFlow 0.8.0 and TensorFlow 0.7.1 is in the MNIST reader, as several TensorFlow functions moved between releases.

Both versions of TensorFlow installed from the provided whl files.  Caffe compiled with ATLAS as BLAS.
