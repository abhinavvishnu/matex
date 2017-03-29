import sys
from mpi4py import MPI
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels, DataSet
# from tensorflow.examples.tutorials.mnist.input_data import extract_images
# from tensorflow.examples.tutorials.mnist.input_data import extract_labels
# from tensorflow.examples.tutorials.mnist.input_data import maybe_download
from tensorflow.models.image.mnist.convolutional import maybe_download
# from tensorflow.examples.tutorials.mnist.input_data import DataSet


def read_data_sets(train_dir, fake_data=False, one_hot=False,
        shuffle=False, validation_percentage=0.1):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        def fake():
            return DataSet([], [], fake_data=True, one_hot=one_hot)
        data_sets.train = fake()
        data_sets.validation = fake()
        data_sets.test = fake()
        return data_sets

    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    WORK_DIRECTORY = 'data'

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    if 0 == rank:
        local_file = maybe_download(TRAIN_IMAGES)
        temp_file = open(local_file, 'r')
        train_images = extract_images(temp_file)
        if shuffle:
            # shuffle the data
            perm = np.arange(train_images.shape[0])
            np.random.shuffle(perm)
            train_images = train_images[perm]
        # bcast the data
        shape = train_images.shape
        shape = comm.bcast(shape, root=0)
        comm.Bcast(train_images, root=0)

        local_file = maybe_download(TRAIN_LABELS)
        temp_file = open(local_file, 'r')
        train_labels = extract_labels(temp_file, one_hot=one_hot)
        if shuffle:
            # shuffle the data, using same indices as images above
            train_labels = train_labels[perm]
        # bcast the data
        shape = train_labels.shape
        shape = comm.bcast(shape, root=0)
        comm.Bcast(train_labels, root=0)

        local_file = maybe_download(TEST_IMAGES)
        temp_file = open(local_file, 'r')
        test_images = extract_images(temp_file)
        shape = test_images.shape
        shape = comm.bcast(shape, root=0)
        comm.Bcast(test_images, root=0)

        local_file = maybe_download(TEST_LABELS)
        temp_file = open(local_file, 'r')
        test_labels = extract_labels(temp_file, one_hot=one_hot)
        shape = test_labels.shape
        shape = comm.bcast(shape, root=0)
        comm.Bcast(test_labels, root=0)
    else:
        shape = None
        shape = comm.bcast(shape, root=0)
        train_images = np.ndarray(shape=shape, dtype=np.uint8)
        comm.Bcast(train_images, root=0)

        shape = None
        shape = comm.bcast(shape, root=0)
        train_labels = np.ndarray(shape=shape)
        comm.Bcast(train_labels, root=0)

        shape = None
        shape = comm.bcast(shape, root=0)
        test_images = np.ndarray(shape=shape, dtype=np.uint8)
        comm.Bcast(test_images, root=0)

        shape = None
        shape = comm.bcast(shape, root=0)
        test_labels = np.ndarray(shape=shape)
        comm.Bcast(test_labels, root=0)

    VALIDATION_SIZE = train_images.shape[0] * validation_percentage
    total = train_images.shape[0] - VALIDATION_SIZE
    count = total / size
    remain = total % size
    if 0 == rank:
        print "total images", total
        print "image subset (%d,%d)=%d" % (total,size,count)
        print "image subset remainder", remain
        
    start = rank * count
    stop = rank * count + count
    if rank < remain:
        start += rank
        stop += rank + 1
    else :
        start += remain
        stop += remain

    VALIDATION_SIZE = int(VALIDATION_SIZE)
    start = int(start)
    stop = int(stop)

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    train_images = train_images[start:stop]
    train_labels = train_labels[start:stop]
    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)
    if 0 == rank:
        print "Rank Start Stop NumExamples"
        sys.stdout.flush()
    for i in xrange(size):
        if rank == i:
            print i,start,stop,data_sets.train.num_examples
            sys.stdout.flush()
        comm.Barrier()
    return data_sets
