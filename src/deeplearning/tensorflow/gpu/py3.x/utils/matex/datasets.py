from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import gzip
import sys
from mpi4py import MPI
import numpy as np
import os
import shutil
import six.moves
import pickle
import tarfile
from itertools import islice
try:
	from pnetcdf import read_pnetcdf
except:
	print("PNetCDF format not configured")
try:
	import h5py
except:
	print("HDF5 format not configured")
	

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class DataSet:
    def __init__(self, data_name, train_batch_size=None, test_batch_size=None, normalize=1.0, train_file=None, validation_file=None, test_file=None,
                 valid_pct=0.0, test_pct=0.0):
        self.dataset = data_name
        self.train_file = train_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.training_data = None
        self.training_labels = None
        self.validation_data = None
        self.validation_labels = None
        self.testing_data = None
        self.testing_labels = None

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        if self.train_batch_size is not None:
            self.train_start = -train_batch_size
            self.train_end = 0
        if self.test_batch_size is not None:
            self.valid_start = -test_batch_size
            self.valid_end = 0
            self.test_start = -test_batch_size
            self.test_end = 0

        if self.dataset.lower() == "mnist":
            self.read_mnist('MNIST_data')
            self.training_data = np.divide(self.training_data, normalize)
            self.validation_data = np.divide(self.validation_data, normalize)
            self.testing_data = np.divide(self.testing_data, normalize)
        elif self.dataset.lower() == "pnetcdf":
            self.training_data, self.training_labels = read_pnetcdf(self.train_file)
            if validation_file:
               self.validation_data, self.valiation_labels = read_pnetcdf(self.validation_file)
            if test_file:
               self.testing_data, self.testing_labels = read_pnetcdf(self.test_file)
        elif self.dataset.lower() == "hdf5":
            self.training_data, self.training_labels = read_hdf5(self.train_file)
            if validation_file:
               self.validation_data, self.valiation_labels = read_hdf5(self.validation_file)
            if test_file:
               self.testing_data, self.testing_labels = read_hdf5(self.test_file)
        elif self.dataset.lower() == "cifar10":
            self.read_cifar10("CIFAR_data", validation_percentage=valid_pct)
        elif self.dataset.lower() == "cifar100":
            self.read_cifar100("CIFAR_data", validation_percentage=valid_pct)
        elif self.dataset.lower() == "csv":
            self.read_csv(self.train_file, validation_percentage=valid_pct, test_percentage=test_pct)

        if type(self.training_labels) is list:
            self.training_labels = self.dense_to_one_hot(self.training_labels, max(self.training_labels) + 1)
        if type(self.validation_labels) is list:
            self.validation_labels = self.dense_to_one_hot(self.validation_labels, max(self.validation_labels) + 1)
        if type(self.testing_labels) is list:
            self.testing_labels = self.dense_to_one_hot(self.testing_labels, max(self.testing_labels) + 1)

    @staticmethod
    def maybe_download(filename, work_directory, source_url):
        if not os.path.exists(work_directory):
            os.makedirs(work_directory)
        filepath = os.path.join(work_directory, filename)
        if not os.path.exists(filepath):
            temp_file_name, _ = six.moves.urllib.request.urlretrieve(source_url + filename, None)
            shutil.copyfile(temp_file_name, filepath)
            file_size = os.path.getsize(filepath)
            print('Successfully downloaded', filename, file_size, 'bytes.')
        return filepath

    def read_mnist(self, train_dir):
        SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

        TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
        TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
        TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
        TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

        if rank == 0:
            train_data_localfile = self.maybe_download(TRAIN_IMAGES, train_dir, SOURCE_URL)
            train_labels_localfile = self.maybe_download(TRAIN_LABELS, train_dir, SOURCE_URL)
            test_data_localfile = self.maybe_download(TEST_IMAGES, train_dir, SOURCE_URL)
            test_labels_localfile = self.maybe_download(TEST_LABELS, train_dir, SOURCE_URL)

            temp_file = open(train_data_localfile, 'rb')
            self.training_data = self.extract_images(temp_file)
            temp_file = open(train_labels_localfile, 'rb')
            self.training_labels = self.extract_labels(temp_file, one_hot=True)
            temp_file = open(test_data_localfile, 'rb')
            self.testing_data = self.extract_images(temp_file)
            temp_file = open(test_labels_localfile, 'rb')
            self.testing_labels = self.extract_labels(temp_file, one_hot=True)
            self.validation_data = self.training_data[50000:]
            self.validation_labels = self.training_labels[50000:]
            self.training_data = self.training_data[:50000]
            self.training_labels = self.training_labels[:50000]
        comm.barrier()
        if rank == 0:
            shape = self.training_data.shape
            comm.bcast(shape, root=0)
            comm.Bcast(self.training_data, root=0)
        else:
            shape = None
            shape = comm.bcast(shape, root=0)
            self.training_data = np.ndarray(shape=shape, dtype=np.uint8)
            comm.Bcast(self.training_data, root=0)
        comm.barrier()
        if rank == 0:
            shape = self.training_labels.shape
            comm.bcast(shape, root=0)
            comm.Bcast(self.training_labels, root=0)
        else:
            shape = None
            shape = comm.bcast(shape, root=0)
            self.training_labels = np.ndarray(shape=shape)
            comm.Bcast(self.training_labels, root=0)
        comm.barrier()
        if rank == 0:
            shape = self.testing_data.shape
            comm.bcast(shape, root=0)
            comm.Bcast(self.testing_data, root=0)
        else:
            shape = None
            shape = comm.bcast(shape, root=0)
            self.testing_data = np.ndarray(shape=shape, dtype=np.uint8)
            comm.Bcast(self.testing_data, root=0)
        comm.barrier()
        if rank == 0:
            shape = self.testing_labels.shape
            comm.bcast(shape, root=0)
            comm.Bcast(self.testing_labels, root=0)
        else:
            shape = None
            shape = comm.bcast(shape, root=0)
            self.testing_labels = np.ndarray(shape=shape)
            comm.Bcast(self.testing_labels, root=0)
        comm.barrier()
        if rank == 0:
            shape = self.validation_data.shape
            comm.bcast(shape, root=0)
            comm.Bcast(self.validation_data, root=0)
        else:
            shape = None
            shape = comm.bcast(shape, root=0)
            self.validation_data = np.ndarray(shape=shape, dtype=np.uint8)
            comm.Bcast(self.validation_data, root=0)
        comm.barrier()
        if rank == 0:
            shape = self.validation_labels.shape
            comm.bcast(shape, root=0)
            comm.Bcast(self.validation_labels, root=0)
        else:
            shape = None
            shape = comm.bcast(shape, root=0)
            self.validation_labels = np.ndarray(shape=shape)
            comm.Bcast(self.validation_labels, root=0)
        comm.barrier()

        total = self.training_data.shape[0]
        count = int(total / size)
        remain = total % size
        if 0 == rank:
            print("total images", total)
            print("image subset (", total, ", ", size, ")=", count, sep='')
            print("image subset remainder", remain)

        start = rank * count
        stop = rank * count + count
        if rank < remain:
            start += rank
            stop += rank + 1
        else:
            start += remain
            stop += remain

        start = int(start)
        stop = int(stop)

        self.training_data = self.training_data[start:stop]
        self.training_labels = self.training_labels[start:stop]

        total = 10000
        count = total / size
        remain = total % size
        small_start = rank * count
        small_stop = rank * count + count
        if rank < remain:
            small_start += rank
            small_stop += rank + 1
        else:
            small_start += remain
            small_stop += remain

        small_start = int(small_start)
        small_stop = int(small_stop)
        self.validation_data = self.validation_data[small_start:small_stop]
        self.validation_labels = self.validation_labels[small_start:small_stop]
        self.testing_data = self.testing_data[small_start:small_stop]
        self.testing_labels = self.testing_labels[small_start:small_stop]

        if 0 == rank:
            print("Rank", "Start", "Stop", "NumExamples")
            sys.stdout.flush()
        for i in range(size):
            if rank == i:
                print(i, start, stop, self.training_data.shape[0])
                sys.stdout.flush()
            comm.Barrier()

    @staticmethod
    def _read32(bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    def extract_images(self, f):
        print('Extracting', f.name)
        data = None
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
            num_images = self._read32(bytestream)
            rows = self._read32(bytestream)
            cols = self._read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols, 1)
        return data

    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def extract_labels(self, f, one_hot=False, num_classes=10):
        labels = None
        print('Extracting', f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2049:
                raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                                 (magic, f.name))
            num_items = self._read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            if one_hot:
                return self.dense_to_one_hot(labels, num_classes)
        return labels

    def read_cifar10(self, train_dir, validation_percentage=0.0):
        self.read_cifar(train_dir, True, validation_percentage)

    def read_cifar100(self, train_dir, validation_percentage=0.0):
        self.read_cifar(train_dir, False, validation_percentage)

    def read_cifar(self, train_dir, is_ten, validation_percentage=0.0):
        if 0 == rank:
            if is_ten:
                self.extract_cifar10(train_dir)
            else:
                self.extract_cifar100(train_dir)
            shape = self.training_data.shape
            comm.bcast(shape, root=0)
            comm.Bcast(self.training_data, root=0)
            shape = self.training_labels.shape
            comm.bcast(shape, root=0)
            comm.Bcast(self.training_labels, root=0)
            shape = self.testing_data.shape
            comm.bcast(shape, root=0)
            comm.Bcast(self.testing_data, root=0)
            shape = self.testing_labels.shape
            comm.bcast(shape, root=0)
            comm.Bcast(self.testing_labels, root=0)
        else:
            shape = None
            shape = comm.bcast(shape, root=0)
            self.training_data = np.ndarray(shape=shape, dtype=np.uint8)
            comm.Bcast(self.training_data, root=0)
            shape = None
            shape = comm.bcast(shape, root=0)
            self.training_labels = np.ndarray(shape=shape)
            comm.Bcast(self.training_labels, root=0)
            shape = None
            shape = comm.bcast(shape, root=0)
            self.testing_data = np.ndarray(shape=shape, dtype=np.uint8)
            comm.Bcast(self.testing_data, root=0)
            shape = None
            shape = comm.bcast(shape, root=0)
            self.testing_labels = np.ndarray(shape=shape)
            comm.Bcast(self.testing_labels, root=0)
        VALIDATION_SIZE = self.training_data.shape[0] * validation_percentage
        total = self.training_data.shape[0] - VALIDATION_SIZE
        count = total / size
        remain = total % size
        if 0 == rank:
            print("total images", total)
            print("image subset (%d,%d)=%d" % (total, size, count))
            print("image subset remainder", remain)

        start = rank * count
        stop = rank * count + count
        if rank < remain:
            start += rank
            stop += rank + 1
        else:
            start += remain
            stop += remain

        VALIDATION_SIZE = int(VALIDATION_SIZE)
        start = int(start)
        stop = int(stop)

        self.validation_data = self.training_data[:VALIDATION_SIZE]
        self.validation_labels = self.training_labels[:VALIDATION_SIZE]
        self.training_data = self.training_data[VALIDATION_SIZE:]
        self.training_labels = self.training_labels[VALIDATION_SIZE:]
        self.training_data = self.training_data[start:stop]
        self.training_labels = self.training_labels[start:stop]
        if 0 == rank:
            print("Rank Start Stop NumExamples")
            sys.stdout.flush()
        for i in range(size):
            if rank == i:
                print(i, start, stop, len(self.training_data))
                sys.stdout.flush()
            comm.Barrier()

    def extract_cifar10(self, dirname):
        filename = 'cifar-10-batches-py'
        filename = os.path.join(dirname, filename)
        CIFAR10 = 'cifar-10-python.tar.gz'
        if not os.path.exists(filename):
            local_file = self.maybe_download(CIFAR10, dirname, 'https://www.cs.toronto.edu/~kriz/')
            if not os.path.exists(filename):
                tarfile.open(local_file).extractall(dirname)
        b1 = self.unpickle(os.path.join(filename, 'data_batch_1'))
        b2 = self.unpickle(os.path.join(filename, 'data_batch_2'))
        b3 = self.unpickle(os.path.join(filename, 'data_batch_3'))
        b4 = self.unpickle(os.path.join(filename, 'data_batch_4'))
        b5 = self.unpickle(os.path.join(filename, 'data_batch_5'))
        test = self.unpickle(os.path.join(filename, 'test_batch'))
        train_images = np.concatenate((b1["data"], b2["data"], b3["data"], b4["data"], b5["data"]))
        train_labels = np.concatenate((b1["labels"], b2["labels"], b3["labels"], b4["labels"], b5["labels"]))
        train_labels = self.dense_to_one_hot(train_labels, 10)
        test_images = test["data"]
        test_labels = np.asarray(test["labels"])
        test_labels = self.dense_to_one_hot(test_labels, 10)
        self.training_data = train_images
        self.training_labels = train_labels
        self.testing_data = test_images
        self.testing_labels = test_labels

    def extract_cifar100(self, dirname):
        CIFAR100 = 'cifar-100-python.tar.gz'
        filename = 'cifar-100-python'
        filename = os.path.join(dirname, filename)
        if not os.path.exists(filename):
            local_file = self.maybe_download(CIFAR100, dirname, 'https://www.cs.toronto.edu/~kriz/')
            if not os.path.exists(filename):
                tarfile.open(local_file).extractall(dirname)
        train = self.unpickle(os.path.join(filename, 'train'))
        test = self.unpickle(os.path.join(filename, 'test'))
        train_images = train["data"]
        train_labels = np.asarray(train["fine_labels"])
        train_labels = self.dense_to_one_hot(train_labels, 100)
        test_images = test["data"]
        test_labels = np.asarray(test["fine_labels"])
        test_labels = self.dense_to_one_hot(test_labels, 100)
        self.training_data = train_images
        self.training_labels = train_labels
        self.testing_data = test_images
        self.testing_labels = test_labels

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            dictionary = pickle.load(fo, encoding='latin1')
        return dictionary

    @staticmethod
    def dense_to_one_hot2(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        labels_one_hot = np.zeros((num_labels, num_classes))
        for i in range(num_labels):
            labels_one_hot[i, int(labels_dense[i])] = 1
        return labels_one_hot

    def read_csv(self, filename, validation_percentage=0.0, test_percentage=0.0):
        line_count = 0
        with open(filename, 'r') as f:
            for _ in f:
                line_count += 1
        even_division = line_count // size
        remainder = line_count % size
        number = even_division
        if rank == 0:
            print("total samples", line_count)
            print("samples subset (%d,%d)=%d" % (line_count, size, even_division))
            print("samples subset remainder", remainder)
        if rank < remainder:
            number += 1
        start = rank * number
        if rank > remainder:
            correction = remainder
        else:
            correction = rank
        start += correction
        finish = start + number
        if rank == 0:
            print("Rank Start Stop NumExamples")
        with open(filename, 'r') as f:
            mydata = np.loadtxt(islice(f, start, finish), delimiter=',')
            print(rank, start, finish, number)
        labels = mydata[:, 0]
        mydata = mydata[:, 1:]
        smallest = int(np.min(labels))
        largest = int(np.max(labels))
        small_list = comm.allgather(smallest)
        smallest = np.min(small_list)
        large_list = comm.allgather(largest)
        largest = np.min(large_list)
        num_classes = 0
        if smallest == 0:
            num_classes = largest - smallest + 1
        elif smallest == -1:
            assert largest == 1
            num_classes = 2
            labels[labels == -1] = 0  # change all -1 values to 0 in labels
        else:
            if 0 == rank:
                print("unrecognized label column in CSV file")
            assert False
        labels = self.dense_to_one_hot2(labels, num_classes)
        if rank == 0:
            print("smallest class, largest class, num_classes")
            print(smallest, largest, num_classes)
        validation_size = int(len(mydata) * validation_percentage)
        test_size = int(len(mydata) * test_percentage)
        if validation_size != 0:
            self.validation_data = mydata[-validation_size:]
            self.validation_labels = labels[-validation_size:]
            mydata = mydata[:-validation_size]
            labels = labels[:-validation_size]
        if test_size != 0:
            self.testing_data = mydata[-test_size:]
            self.testing_labels = labels[-test_size:]
            mydata = mydata[:-test_size]
            labels = labels[:-test_size]
        self.training_data = mydata
        self.training_labels = labels

    def read_hdf5(self, filename):
        f = h5py.File(filename, 'r')
        names = [name for name in f]
        items_per_class = []
        for name in names:
            items_per_class.append(f[name].shape[0])
        even_division = [x//size for x in items_per_class]
        remainder = [x % size for x in items_per_class]
        number = even_division
        if rank == 0:
            print("total samples", items_per_class)
            print("samples subset (", items_per_class, ",", size, ")=", even_division, sep='')
            print("samples subset remainder", remainder)
        r = 0
        while remainder != [0 for x in remainder]:
            for i in range(len(remainder)):
                if remainder[i] != 0:
                    remainder[i] -= 1
                    if rank == r:
                        number[i] += 1
                    r += 1
        start = [rank * x for x in number]
        correction = [0] * len(remainder)
        for i in range(len(remainder)):
            if rank > remainder[i]:
                correction[i] = remainder[i]
            else:
                correction[i] = rank
        start = [start[i] + correction[i] for i in range(len(start))]
        finish = [start[i] + number[i] for i in range(len(start))]
        if rank == 0:
            print("Rank Start Stop NumExamples")
        print(rank, start, finish, number)
		
        tup = [f[names[i]][start[i]:finish[i]] for i in range(len(names))]
		
        train_labels = [i for i in range(len(names)) for j in range(number[i])]
        train_data = np.concatenate(tup)
		
        rng_state = np.random.get_state()
        np.random.shuffle(train_data)
        np.random.set_state(rng_state)
        np.random.shuffle(train_labels)
        train_labels = self.dense_to_one_hot2(train_labels, len(names))
        return train_data, train_labels
		
    def next_train_batch(self):
        if self.train_batch_size is None:
            raise UserWarning("No Training Batch Size Specified")
        self.train_start += self.train_batch_size
        self.train_end += self.train_batch_size
        if self.train_start > len(self.training_data):
            self.train_start %= len(self.training_data)
            self.train_end %= len(self.training_data)
            train_batch = self.training_data[self.train_start:self.train_end]
            train_batch_labels = self.training_labels[self.train_start:self.train_end]
        elif self.train_end > len(self.training_data):
            train_batch = self.training_data[self.train_start:self.train_end]
            train_batch_labels = self.training_labels[self.train_start:self.train_end]
            self.train_end %= len(self.training_data)
            train2 = self.training_data[:self.train_end]
            train_label2 = self.training_labels[:self.train_end]
            train_batch = np.concatenate((train_batch, train2))
            train_batch_labels = np.concatenate((train_batch_labels, train_label2))
        else:
            train_batch = self.training_data[self.train_start:self.train_end]
            train_batch_labels = self.training_labels[self.train_start:self.train_end]
        return [train_batch, train_batch_labels]

    def next_validation_batch(self):
        if self.test_batch_size is None:
            raise UserWarning("No Testing Batch Size Specified")
        self.valid_start += self.test_batch_size
        self.valid_end += self.test_batch_size
        if self.valid_start > len(self.validation_data):
            self.valid_start %= len(self.validation_data)
            self.valid_end %= len(self.validation_data)
            validation_batch = self.validation_data[self.valid_start:self.valid_end]
            validation_batch_labels = self.validation_labels[self.valid_start:self.valid_end]
        elif self.valid_end > len(self.validation_data):
            validation_batch = self.validation_data[self.valid_start:self.valid_end]
            validation_batch_labels = self.validation_labels[self.valid_start:self.valid_end]
            self.valid_end %= len(self.validation_data)
            validation2 = self.validation_data[:self.valid_end]
            validation_label2 = self.validation_labels[:self.valid_end]
            validation_batch = np.concatenate((validation_batch, validation2))
            validation_batch_labels = np.concatenate((validation_batch_labels, validation_label2))
        else:
            validation_batch = self.validation_data[self.valid_start:self.valid_end]
            validation_batch_labels = self.validation_labels[self.valid_start:self.valid_end]
        return [validation_batch, validation_batch_labels]

    def next_test_batch(self):
        if self.test_batch_size is None:
            raise UserWarning("No Testing Batch Size Specified")
        self.test_start += self.test_batch_size
        self.test_end += self.test_batch_size
        if self.test_start > len(self.testing_data):
            self.test_start %= len(self.testing_data)
            self.test_end %= len(self.testing_data)
            test_batch = self.testing_data[self.test_start:self.test_end]
            test_batch_labels = self.testing_labels[self.test_start:self.test_end]
        elif self.test_end > len(self.testing_data):
            test_batch = self.testing_data[self.test_start:self.test_end]
            test_batch_labels = self.testing_labels[self.test_start:self.test_end]
            self.test_end %= len(self.testing_data)
            test2 = self.testing_data[:self.test_end]
            test_label2 = self.testing_labels[:self.test_end]
            test_batch = np.concatenate((test_batch, test2))
            test_batch_labels = np.concatenate((test_batch_labels, test_label2))
        else:
            test_batch = self.testing_data[self.test_start:self.test_end]
            test_batch_labels = self.testing_labels[self.test_start:self.test_end]
        return [test_batch, test_batch_labels]

if __name__ == '__main__':
    mnist = DataSet("MNIST")
    odata = mnist.training_data
    olabels = mnist.training_labels
    data = comm.gather(odata, root=0)
    labels = comm.gather(olabels, root=0)
    if rank == 0:
       data = np.reshape(data, [-1, np.shape(data)[-3], np.shape(data)[-2], np.shape(data)[-1]])
       labels = np.reshape(labels, [-1, np.shape(labels)[-1]])
       tmp = np.zeros([50000])
       for i in range(50000):
           tmp[i] = np.argmax(labels[i])
       labels = tmp
       data = np.reshape(data, [50000, 784])
       labels = np.expand_dims(labels, 1)
       temp = np.concatenate((labels, data), axis=1)
       np.savetxt("mnist.csv", temp, delimiter=",")
       print('CSV Generated')
    DataSet("CIFAR10")
    DataSet("CIFAR100")
    DataSet("CSV", train_file="mnist.csv", valid_pct=0.0, test_pct=0.0)
    DataSet("PNETCDF", train_file="mnist_train.nc")
