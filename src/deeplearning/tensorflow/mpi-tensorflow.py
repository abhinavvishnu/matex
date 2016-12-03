from mpi4py import MPI
from mnist_mpi_reader import read_data_sets
from cifar_mpi_reader import read_cifar10
from cifar_mpi_reader import read_cifar100
from csv_mpi_reader import read_csv
from pnetcdf import read_pnetcdf
import tensorflow as tf
import numpy as np
import time
import sys
import resource
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

epoch = 0

parser = argparse.ArgumentParser()
parser.add_argument('--conv_layers', type=int, default=None, nargs='+', help='number of conv layers (space separated)')
parser.add_argument('--full_layers', type=int, default=[30], nargs='+', help='number of full layers (space separated)')
parser.add_argument('--train_batch', type=int, default=10, help='training batch size')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.1, nargs='+', help='learning rate')
parser.add_argument('--input_shape', type=int, default=[784], nargs='+', help='input shape')
parser.add_argument('--output_size', type=int, default=10, help='number of classes')
parser.add_argument('--data', type=str, default="MNIST", help='dataset')
parser.add_argument('--filename', type=str, default=None, help='filename')
parser.add_argument('--filename2', type=str, default=None, help='filename')
parser.add_argument('--valid_pct', type=float, default=0.1, help='validation percent')
parser.add_argument('--test_pct', type=float, default=0.1, help='testing percent')
parser.add_argument('--time', type=float, default=-1, help='training time')
parser.add_argument('--threads', type=int, default=0, help='number of threads')
parser.add_argument('--inter_threads', type=int, default=0, help='number of internal threads')
parser.add_argument('--intra_threads', type=int, default=0, help='number of intra-op threads')
parser.add_argument('--error_batch', action='store_true', help='batch the error calculation')
parser.add_argument('--top', type=int, default=1, help='top')
args = parser.parse_args()

if args.inter_threads == 0:
    args.inter_threads = args.threads
if args.intra_threads == 0:
    args.intra_threads = args.threads

# data parsing

max_accuracy_encountered_value = 0
max_accuracy_encountered_epoch = 0
max_accuracy_encountered_time = 0
full_dat = None
full_lab = None
valid_dat = None
valid_lab = None
test_dat = None
test_lab = None
if args.data == "MNIST":
    mnist = read_data_sets('MNIST_data', one_hot=True, validation_percentage=args.valid_pct)
    full_dat = mnist.train._images
    full_lab = mnist.train._labels
    valid_dat = mnist.validation.images
    valid_lab = mnist.validation.labels
    test_dat = mnist.test.images
    test_lab = mnist.test.labels
    args.input_shape = [784]
    if args.conv_layers:
        args.input_shape = [28, 28, 1]
    args.output_size = 10
elif args.data == "CIFAR10":
    cifar10 = read_cifar10("CIFAR_data", one_hot=True, validation_percentage=args.valid_pct)
    full_dat = cifar10.train._images
    full_lab = cifar10.train._labels
    valid_dat = cifar10.validation.images
    valid_lab = cifar10.validation.labels
    test_dat = cifar10.test.images
    test_lab = cifar10.test.labels
    if args.conv_layers:
        args.input_shape = [32, 32, 3]
    else:
        args.input_shape = [full_dat.shape[1]]
    args.output_size = 10
elif args.data == "CIFAR100":
    cifar100 = read_cifar100("CIFAR_data", one_hot=True, validation_percentage=args.valid_pct)
    full_dat = cifar100.train._images
    full_lab = cifar100.train._labels
    valid_dat = cifar100.validation.images
    valid_lab = cifar100.validation.labels
    test_dat = cifar100.test.images
    test_lab = cifar100.test.labels
    if args.conv_layers:
        args.input_shape = [32, 32, 3]
    else:
        args.input_shape = [full_dat.shape[1]]
    args.output_size = 100
elif args.data == "CSV":
    full_dat, full_lab, valid_dat, valid_lab, test_dat, test_lab = read_csv(args.filename,
                                                                            validation_percentage=args.valid_pct,
                                                                            test_percentage=args.test_pct)
    args.input_shape = [full_dat.shape[1]]
    args.output_size = full_lab.shape[1]
elif args.data == "PNETCDF":
    training_data, training_labels = read_pnetcdf(args.filename)
    testing_data, testing_labels = read_pnetcdf(args.filename2)
    args.input_shape = training_data.shape[1:]
    args.output_size = np.max(training_labels)
    if 0 in training_labels:
        args.output_size += 1

if 0 == rank:
    print full_dat.shape

input_size = np.prod(args.input_shape)

# set up network


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def create_conv_layer(features_in, features_out, layer_list, weight_list, bias_list):
    weight_list.append(weight_variable([5, 5, features_in, features_out]))
    bias_list.append(bias_variable([features_out]))
    temp_w = len(weight_list)
    temp_b = len(bias_list)
    temp_l = len(layer_list)
    h_conv1 = tf.nn.relu(conv2d(layer_list[temp_l - 1], weight_list[temp_w - 1]) + bias_list[temp_b - 1])
    h_pool1 = max_pool_2x2(h_conv1)
    layer_list.append(h_pool1)


def create_full_layer(in_size, out_size, layer_list, weight_list, bias_list):
    weight_list.append(tf.Variable(tf.random_normal([in_size, out_size], stddev=1.0 / in_size)))
    bias_list.append(tf.Variable(tf.random_normal([out_size], stddev=1.0 / in_size)))
    temp_w = len(weight_list)
    temp_b = len(bias_list)
    temp_l = len(layer_list)
    layer_list.append(tf.nn.sigmoid(tf.matmul(layer_list[temp_l - 1], weight_list[temp_w - 1]) + bias_list[temp_b - 1]))


time_global_start = time.time()


def mpi_average(tensor):
    comm.Allreduce(MPI.IN_PLACE, tensor, MPI.SUM)
    tensor /= size
    return tensor


def populate_graph(args_in):
    weights = []
    biases = []
    layers = []

    x = tf.placeholder(tf.float32, [None, input_size])
    y_ = tf.placeholder(tf.float32, [None, args_in.output_size])

    layers.append(x)
    layers.append(tf.reshape(x, [-1] + args_in.input_shape))

    if args_in.conv_layers:
        args_in.conv_layers = [args_in.input_shape[-1]] + args_in.conv_layers
        for i in range(len(args_in.conv_layers) - 1):
            create_conv_layer(args_in.conv_layers[i], args_in.conv_layers[i + 1], layers, weights, biases)
        transition_size = args_in.input_shape[0] * args_in.input_shape[1] / 4 ** (len(args_in.conv_layers) - 1)
        layers.append(tf.reshape(layers[len(args_in.conv_layers)], [-1, transition_size * args_in.conv_layers[-1]]))
        create_full_layer(transition_size * args_in.conv_layers[-1], args_in.full_layers[0], layers, weights, biases)
    if not args_in.conv_layers:
        args_in.full_layers = [input_size] + args_in.full_layers
    for i in range(len(args_in.full_layers) - 1):
        create_full_layer(args_in.full_layers[i], args_in.full_layers[i + 1], layers, weights, biases)
    W = tf.Variable(tf.random_normal([args_in.full_layers[-1], args_in.output_size],
                                     stddev=1.0 / args_in.full_layers[-1]))
    b = tf.Variable(tf.random_normal([args_in.output_size], stddev=1.0 / args_in.full_layers[-1]))
    weights.append(W)
    biases.append(b)

    w_holder = [tf.placeholder(tf.float32, w.get_shape()) for w in weights]
    b_holder = [tf.placeholder(tf.float32, b.get_shape()) for b in biases]
    w_assign = [w.assign(p) for w, p in zip(weights, w_holder)]
    b_assign = [b.assign(p) for b, p in zip(biases, b_holder)]

    y = tf.nn.softmax(tf.matmul(layers[-1], W) + b)

    cross_entropy = - tf.reduce_mean(tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=1))

    # optimizer = tf.train.AdagradOptimizer(args_in.learning_rate)
    # optimizer = tf.train.MomentumOptimizer(args_in.learning_rate, 0.9)
    optimizer = tf.train.GradientDescentOptimizer(args_in.learning_rate)

    grads_and_vars = optimizer.compute_gradients(cross_entropy, weights + biases)
    grads_and_vars = [(tf.py_func(mpi_average, [gv[0]], tf.float32), gv[1]) for gv in grads_and_vars]
    train_step = optimizer.apply_gradients(grads_and_vars)

    init = tf.global_variables_initializer()
    sess = tf.Session(
        config=tf.ConfigProto(
            inter_op_parallelism_threads=args_in.inter_threads,
            intra_op_parallelism_threads=args_in.intra_threads))

    correct = tf.nn.in_top_k(y, tf.argmax(y_, 1), args_in.top)
    accuracy = tf.reduce_sum(tf.cast(correct, tf.int32))

    sess.run(init)

    r_weights = sess.run(weights)
    r_biases = sess.run(biases)
    for i in range(len(r_weights)):
        r_weights[i] = comm.bcast(r_weights[i], root=0)
    for i in range(len(r_biases)):
        r_biases[i] = comm.bcast(r_biases[i], root=0)

    feed_dict = {}
    for d, p in zip(r_weights, w_holder):
        feed_dict[p] = d
    for d, p in zip(r_biases, b_holder):
        feed_dict[p] = d
    sess.run(w_assign + b_assign, feed_dict=feed_dict)

    ops = {
        "sess": sess,
        "x": x,
        "y_": y_,
        "weights": weights,
        "biases": biases,
        "w_holder": w_holder,
        "b_holder": b_holder,
        "w_assign": w_assign,
        "b_assign": b_assign,
        "train_step": train_step,
        "cross_entropy": cross_entropy,
        "accuracy": accuracy,
    }

    return ops


def run_graph(data, labels, ops, args_in):
    global time_global_start
    global epoch
    global max_accuracy_encountered_value
    global max_accuracy_encountered_epoch
    global max_accuracy_encountered_time

    time_epoch_start = time.time()
    time_comm = 0.0

    sess = ops["sess"]
    x = ops["x"]
    y_ = ops["y_"]

    train_step = ops["train_step"]
    cross_entropy = ops["cross_entropy"]
    accuracy = ops["accuracy"]

    number_of_batches = len(data) / args_in.train_batch
    if number_of_batches == 0:
        number_of_batches = 1

    for i in range(number_of_batches):
        lo = i * args_in.train_batch
        hi = (i + 1) * args_in.train_batch
        batch_xs = data[lo:hi]
        batch_ys = labels[lo:hi]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    sum_error = 0.0
    if args_in.error_batch:
        for i in range(number_of_batches):
            lo = i * args_in.train_batch
            hi = (i + 1) * args_in.train_batch
            batch_xs = data[lo:hi]
            batch_ys = labels[lo:hi]
            sum_error += sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
    else:
        sum_error = sess.run(cross_entropy, feed_dict={x: data, y_: labels})
    time_this = time.time()
    sum_error_all = comm.allreduce(sum_error) / size
    time_comm += time.time() - time_this
    accurate = 0.0
    if args_in.error_batch:
        test_batch_count = len(test_dat) / args_in.train_batch
        if test_batch_count == 0:
            test_batch_count = 1
        for i in range(test_batch_count):
            lo = i * args_in.train_batch
            hi = (i + 1) * args_in.train_batch
            batch_xs = test_dat[lo:hi]
            batch_ys = test_lab[lo:hi]
            accurate += sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    else:
        accurate = sess.run(accuracy, feed_dict={x: test_dat, y_: test_lab})
    time_this = time.time()
    accurate = comm.allreduce(accurate, MPI.SUM)
    acc_count = comm.allreduce(len(test_dat), MPI.SUM)
    accurate = float(accurate) / acc_count
    time_comm += time.time() - time_this

    if accurate > max_accuracy_encountered_value:
        max_accuracy_encountered_value = accurate
        max_accuracy_encountered_epoch = epoch
        max_accuracy_encountered_time = time.time() - time_global_start

    time_all = time.time() - time_epoch_start

    if 0 == rank:
        print "%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % (
            epoch + 1,
            time_all,
            time.time() - time_global_start,
            accurate,
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.0,
            time_comm / time_all,
            sum_error_all,
        )
    sys.stdout.flush()

if 0 == rank:
    print "epoch,etime,ctime,accuracy,MB_mem,time_comm,error"

data_threshold = int(len(full_dat) / 2)
active_dat = full_dat
active_lab = full_lab
inactive_dat = np.empty([0] + list(full_dat.shape[1:]), full_dat.dtype)
inactive_lab = np.empty([0] + list(full_lab.shape[1:]), full_lab.dtype)

if args.time > 0:
    ops = populate_graph(args)
    while args.time > (time.time() - time_global_start):
        run_graph(active_dat, active_lab, ops, args)
        epoch += 1
else:
    ops = populate_graph(args)
    for epoch in range(args.epochs):
        run_graph(active_dat, active_lab, ops, args)

if 0 == rank:
    print "max accuracy achieved value", max_accuracy_encountered_value
    print "max accuracy achieved epoch", max_accuracy_encountered_epoch + 1
    print "max accuracy achieved time", max_accuracy_encountered_time
