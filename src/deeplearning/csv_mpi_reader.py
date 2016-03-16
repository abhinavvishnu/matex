from itertools import islice

from mpi4py import MPI
import numpy as np

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def dense_to_one_hot2(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((num_labels,num_classes))
    for i in xrange(num_labels):
        labels_one_hot[i,int(labels_dense[i])] = 1
    return labels_one_hot


def read_csv(filename, line_count=-1,
        validation_percentage=0.1, test_percentage=0.1):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mydata = None
    validdata = None
    testdata = None
    if 0 == rank:
        if line_count == -1:
            line_count = 0
            for line in open(filename):
                line_count += 1
        valid_size = line_count * validation_percentage
        test_size = line_count * test_percentage
        total = line_count - valid_size - test_size
        assert total > size
        count = total / size
        remain = total % size
        print "total samples", total
        print "samples subset (%d,%d)=%d" % (total,size,count)
        print "samples subset remainder", remain
        infile = open(filename)
        mycount = count
        if 0 < remain:
            mycount += 1
        mydata = np.loadtxt(islice(infile, mycount), delimiter=',')
        print "rank",rank,"got its data"
        for i in xrange(1,size):
            mycount = count
            if i < remain:
                mycount += 1
            theirdata = np.loadtxt(islice(infile, mycount), delimiter=',')
            print "rank",rank,"loaded data for",i
            comm.send(theirdata.shape, dest=i, tag=11)
            comm.Send(theirdata, dest=i, tag=77)
        validdata = np.loadtxt(islice(infile, valid_size), delimiter=',')
        shape = comm.bcast(validdata.shape, root=0)
        comm.Bcast(validdata, root=0)
        testdata = np.loadtxt(islice(infile, test_size), delimiter=',')
        shape = comm.bcast(testdata.shape, root=0)
        comm.Bcast(testdata, root=0)
    else:
        shape = comm.recv(source=0, tag=11)
        mydata = np.empty(shape)
        comm.Recv(mydata, source=0, tag=77)
        shape = comm.bcast(None, root=0)
        validdata = np.empty(shape)
        comm.Bcast(validdata, root=0)
        shape = comm.bcast(None, root=0)
        testdata = np.empty(shape)
        comm.Bcast(testdata, root=0)
        print "rank",rank,"got its data"
    # first column is the classification label, assumed to be 0/1 or -1/1
    labels = mydata[:,0]
    mydata = mydata[:,1:]
    smallest_ = np.min(labels)
    largest_ = np.max(labels)
    smallest = np.empty(1)
    largest = np.empty(1)
    comm.Allreduce(smallest_, smallest, MPI.MIN)
    comm.Allreduce(largest_, largest, MPI.MAX)
    smallest = smallest[0]
    largest = largest[0]
    if smallest == 0:
        num_classes = largest - smallest + 1
    elif smallest == -1:
        assert largest == 1
        num_classes = 2
        labels[labels==-1] = 0 # change all -1 values to 0 in labels
    else:
        if 0 == rank:
            print "unrecognized label column in CSV file"
        assert False

    if 0 == rank:
        print labels.shape
        print labels[:,None].shape
        print mydata.shape
        print "smallest class, largest class, num_classes"
        print smallest, largest, num_classes
    comm.Barrier()

    labels = dense_to_one_hot2(labels[:,None], num_classes)

    validlabels = validdata[:,0]
    validdata = validdata[:,1:]
    if smallest == -1:
        validlabels[validlabels==-1] = 0 # change all -1 values to 0 in labels
    validlabels = dense_to_one_hot2(validlabels, num_classes)

    testlabels = testdata[:,0]
    testdata = testdata[:,1:]
    if smallest == -1:
        testlabels[testlabels==-1] = 0 # change all -1 values to 0 in labels
    testlabels = dense_to_one_hot2(testlabels, num_classes)

    # normalize the data per attribute -- gross approximation
    for col in xrange(mydata.shape[1]):
        stddev = None
        mean = None
        if 0 == rank:
            stddev = mydata[:,col].std()
            mean = mydata[:,col].mean()
        stddev = comm.bcast(stddev)
        mean = comm.bcast(mean)
        if abs(stddev) > 0.0:
            mydata[:,col] -= mean
            mydata[:,col] /= stddev
            validdata[:,col] -= mean
            validdata[:,col] /= stddev
            testdata[:,col] -= mean
            testdata[:,col] /= stddev

    return mydata,labels,validdata,validlabels,testdata,testlabels
