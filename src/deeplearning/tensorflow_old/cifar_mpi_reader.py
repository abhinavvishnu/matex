"""Functions for downloading and reading CIFAR data."""
from __future__ import absolute_import
from __future__ import division

import os
import sys

from mpi4py import MPI
import numpy as np
from six.moves import urllib

SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/'
CIFAR10 = 'cifar-10-python.tar.gz'
CIFAR100 = 'cifar-100-python.tar.gz'
filename_to_untarred_name = {
    CIFAR10 : 'cifar-10-batches-py',
    CIFAR100: 'cifar-100-python'
}

class DataSet(object):

  def __init__(self, images, labels):
    """Construct a DataSet."""

    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape,
                                               labels.shape))
    self._num_examples = images.shape[0]

    # Convert from [0, 255] -> [0.0, 1.0].
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def maybe_download(filename, work_directory):
  """Download the data from Krizhevsky's website, unless it's already here."""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print 'Successfully downloaded', filename, statinfo.st_size, 'bytes.'
  return filepath

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_cifar10(dirname):
    filename = filename_to_untarred_name[CIFAR10]
    filename = os.path.join(dirname, filename)
    if not os.path.exists(filename):
        local_file = maybe_download(CIFAR10, dirname)
        if not os.path.exists(filename):
            import tarfile
            tar = tarfile.open(local_file).extractall(dirname)
    b1 = unpickle(os.path.join(filename, 'data_batch_1'))
    b2 = unpickle(os.path.join(filename, 'data_batch_2'))
    b3 = unpickle(os.path.join(filename, 'data_batch_3'))
    b4 = unpickle(os.path.join(filename, 'data_batch_4'))
    b5 = unpickle(os.path.join(filename, 'data_batch_5'))
    test = unpickle(os.path.join(filename, 'test_batch'))
    train_images = np.concatenate((b1["data"], b2["data"], b3["data"], b4["data"], b5["data"]))
    train_labels = np.concatenate((b1["labels"], b2["labels"], b3["labels"], b4["labels"], b5["labels"]))
    train_labels = dense_to_one_hot(train_labels, 10)
    test_images = test["data"]
    test_labels = np.asarray(test["labels"])
    test_labels = dense_to_one_hot(test_labels, 10)
    return train_images,train_labels,test_images,test_labels

def extract_cifar100(dirname):
    filename = filename_to_untarred_name[CIFAR100]
    filename = os.path.join(dirname, filename)
    if not os.path.exists(filename):
        local_file = maybe_download(CIFAR100, dirname)
        if not os.path.exists(filename):
            import tarfile
            tar = tarfile.open(local_file).extractall(dirname)
    train = unpickle(os.path.join(filename, 'train'))
    test = unpickle(os.path.join(filename, 'test'))
    train_images = train["data"]
    train_labels = np.asarray(train["fine_labels"])
    train_labels = dense_to_one_hot(train_labels, 100)
    test_images = test["data"]
    test_labels = np.asarray(test["fine_labels"])
    test_labels = dense_to_one_hot(test_labels, 100)
    return train_images,train_labels,test_images,test_labels
    
def read_cifar(train_dir, is_ten, one_hot=False, shuffle=False,
        validation_percentage=0.1):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    class DataSets(object):
        pass
    data_sets = DataSets()
    if 0 == rank:
        if is_ten:
            train_images,train_labels,test_images,test_labels = extract_cifar10(train_dir)
        else:
            train_images,train_labels,test_images,test_labels = extract_cifar100(train_dir)
        if shuffle:
            # shuffle the data
            perm = np.arange(train_images.shape[0])
            np.random.shuffle(perm)
            train_images = train_images[perm]
            train_labels = train_labels[perm]
        shape = train_images.shape
        shape = comm.bcast(shape, root=0)
        comm.Bcast(train_images, root=0)
        shape = train_labels.shape
        shape = comm.bcast(shape, root=0)
        comm.Bcast(train_labels, root=0)
        shape = test_images.shape
        shape = comm.bcast(shape, root=0)
        comm.Bcast(test_images, root=0)
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

def read_cifar10(train_dir, one_hot=False, shuffle=False,
        validation_percentage=0.1):
    return read_cifar(train_dir, True, one_hot, shuffle,
            validation_percentage)

def read_cifar100(train_dir, one_hot=False, shuffle=False,
        validation_percentage=0.1):
    return read_cifar(train_dir, False, one_hot, shuffle,
            validation_percentage)

