from mpi4py import MPI
from mnist_mpi_reader import read_data_sets
from cifar_mpi_reader import read_cifar10
from cifar_mpi_reader import read_cifar100
from csv_mpi_reader import read_csv
import tensorflow as tf
import numpy as np
import time
import math
import sys
import getopt
import resource

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

opts,args = getopt.getopt(sys.argv[1:], "a:b:c:d:e:fg:i:hj:k:l:t:r:s:z:y",
["conv_layers=",
"full_layers=",
"train_batch=",
"epochs=",
"learning_rate=",
"help",
"input_shape=",
"output_size=",
"data=",
"filename=",
"valid_pct=",
"test_pct=",
"time=",
"threads=",
"inter_threads=",
"intra_threads=",
"error_batch",
"merge_every=",
"top=",
])

conv_layers = []
full_layers = [30]
train_batch = 10
epochs = 30
learning_rate = 0.1
data = "MNIST"
input_shape = [784]
threads = 1
inter_threads = 0
intra_threads = 0
filename = None
valid_pct = 0.1
test_pct = 0.1
stop_time = -1
epoch = 0
error_batch = False
merge_every = 0
top = 1

help_str = ""
help_str +=   "    --conv_layers"
help_str += "\n    --full_layers"
help_str += "\n    --train_batch"
help_str += "\n    --epochs"
help_str += "\n    --learning_rate"
help_str += "\n    --help"
help_str += "\n    --input_shape"
help_str += "\n    --output_size"
help_str += "\n    --data"
help_str += "\n    --threads"
help_str += "\n    --inter_threads"
help_str += "\n    --intra_threads"
help_str += "\n    --filename"
help_str += "\n    --valid_pct"
help_str += "\n    --test_pct"
help_str += "\n    --time"
help_str += "\n    --error_batch"
help_str += "\n    --merge_every"
help_str += "\n    --top"

for opt,arg in opts:
    if opt == '-a' or opt == '--conv_layers':
        temp = arg.split(",")
        for i in range(len(temp)):
            temp[i] = int(temp[i])
        conv_layers = temp
    elif opt == '-b' or opt == '--full_layers':
        temp = arg.split(",")
        for i in range(len(temp)):
            temp[i] = int(temp[i])
        full_layers = temp
    elif opt == '-c' or opt == '--train_batch':
        train_batch = int(arg)
    elif opt == '-d' or opt == '--epochs':
        epochs = int(arg)
    elif opt == '-e' or opt == '--learning_rate':
        learning_rate = float(arg)
    elif opt == '-h' or opt == '--help':
        if 0 == rank:
            print help_str
        sys.exit()
    elif opt == '-j' or opt == '--input_shape':
        temp = arg.split(",")
        for i in range(len(temp)):
            temp[i] = int(temp[i])
        input_shape = temp
    elif opt == '-k' or opt == '--output_size':
        output_size = int(arg)
    elif opt == '-l' or opt == '--data':
        data = arg
    elif opt == '-t' or opt == '--threads':
        threads = int(arg)
    elif opt == '--inter_threads':
        inter_threads = int(arg)
    elif opt == '--intra_threads':
        intra_threads = int(arg)
    elif opt == '--filename':
        filename = arg
    elif opt == '--valid_pct':
        valid_pct = float(arg)
    elif opt == '--test_pct':
        test_pct = float(arg)
    elif opt == '--time':
        stop_time = float(arg)
    elif opt == '--error_batch':
        error_batch = True
    elif opt == '--merge_every':
        merge_every = int(arg)
    elif opt == '--top':
        top = int(arg)

if 0 == inter_threads:
    inter_threads = threads
if 0 == intra_threads:
    intra_threads = threads

#data parsing

max_accuracy_encountered_value = 0
max_accuracy_encountered_epoch = 0
max_accuracy_encountered_time = 0
full_dat = None
full_lab = None
valid_dat = None
valid_lab = None
test_dat = None
test_lab = None
if data == "MNIST":
    mnist = read_data_sets('MNIST_data', one_hot=True, validation_percentage=valid_pct)
    full_dat = mnist.train._images
    full_lab = mnist.train._labels
    valid_dat = mnist.validation.images
    valid_lab = mnist.validation.labels
    test_dat = mnist.test.images
    test_lab = mnist.test.labels
    if conv_layers != []:
        input_shape = [28, 28, 1]
    output_size = 10
elif data == "CIFAR10":
    cifar10 = read_cifar10("CIFAR_data", one_hot=True,
            validation_percentage=valid_pct)
    full_dat = cifar10.train._images
    full_lab = cifar10.train._labels
    valid_dat = cifar10.validation.images
    valid_lab = cifar10.validation.labels
    test_dat = cifar10.test.images
    test_lab = cifar10.test.labels
    if conv_layers != []:
        input_shape = [32, 32, 3]
    else:
        input_shape = [full_dat.shape[1]]
    output_size = 10
elif data == "CIFAR100":
    cifar100 = read_cifar100("CIFAR_data", one_hot=True,
            validation_percentage=valid_pct)
    full_dat = cifar100.train._images
    full_lab = cifar100.train._labels
    valid_dat = cifar100.validation.images
    valid_lab = cifar100.validation.labels
    test_dat = cifar100.test.images
    test_lab = cifar100.test.labels
    if conv_layers != []:
        input_shape = [32, 32, 3]
    else:
        input_shape = [full_dat.shape[1]]
    output_size = 100
elif data == "CSV":
    full_dat,full_lab,valid_dat,valid_lab,test_dat,test_lab = read_csv(filename,
            validation_percentage=valid_pct,
            test_percentage=test_pct)
    input_shape = [full_dat.shape[1]]
    output_size = full_lab.shape[1]

if 0 == rank:
    print full_dat.shape

input_size = 1
for i in input_shape:
    input_size *= i

#set up network

def weight_variable(shape, saved_state, index):
    if saved_state is None:
        initial = tf.truncated_normal(shape, stddev=0.1)
    else:
        initial = saved_state[0][index]
    return tf.Variable(initial)

def bias_variable(shape, saved_state, index):
    if saved_state is None:
        initial = tf.constant(0.1, shape=shape)
    else:
        initial = saved_state[1][index]
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def create_conv_layer(features_in, features_out, layer_list,
        weight_list, bias_list, saved_state):
    index = len(weight_list)
    weight_list.append(weight_variable([5, 5, features_in, features_out],
        saved_state, index))
    bias_list.append(bias_variable([features_out], saved_state, index))
    temp_w = len(weight_list)
    temp_b = len(bias_list)
    temp_l = len(layer_list)
    h_conv1 = tf.nn.relu(conv2d(layer_list[temp_l-1],
                         weight_list[temp_w-1]) + bias_list[temp_b-1])
    h_pool1 = max_pool_2x2(h_conv1)
    layer_list.append(h_pool1)

def create_full_layer(in_size, out_size, layer_list, weight_list,
        bias_list, saved_state):
    if saved_state is None:
        weight_list.append(tf.Variable(
            tf.random_normal([in_size, out_size], stddev=1.0/in_size)))
        bias_list.append(tf.Variable(
            tf.random_normal([out_size], stddev=1.0/in_size)))
    else:
        index = len(weight_list)
        weight_list.append(tf.Variable(saved_state[0][index]))
        bias_list.append(tf.Variable(saved_state[1][index]))
    temp_w = len(weight_list)
    temp_b = len(bias_list)
    temp_l = len(layer_list)
    layer_list.append(tf.nn.sigmoid(tf.matmul(layer_list[temp_l-1], weight_list[temp_w-1])+bias_list[temp_b-1]))

time_global_start = time.time()


def populate_graph(
            conv_layers,
            full_layers,
            learning_rate,
            input_shape,
            saved_state):

    weights = []
    biases = []
    layers = []
    
    x = tf.placeholder(tf.float32, [None, input_size])
    y_ = tf.placeholder(tf.float32, [None, output_size])
    
    layers.append(x)
    layers.append(tf.reshape(x, [-1]+input_shape))
    
    if conv_layers != []:
        conv_layers = [input_shape[-1]] + conv_layers
        for i in range(len(conv_layers)-1):
            create_conv_layer(conv_layers[i], conv_layers[i+1], layers,
                    weights, biases, saved_state)
        transition_size = input_shape[0]*input_shape[1]/4**(len(conv_layers)-1)
        layers.append(tf.reshape(layers[len(conv_layers)], [-1, transition_size*conv_layers[-1]]))
        create_full_layer(transition_size * conv_layers[-1],
                full_layers[0], layers, weights, biases, saved_state)
    if conv_layers == []:
        full_layers = [input_size] + full_layers
    for i in range(len(full_layers)-1):
        create_full_layer(full_layers[i], full_layers[i+1], layers,
                weights, biases, saved_state)
    if saved_state is None:
        W = tf.Variable(tf.random_normal([full_layers[-1], output_size], stddev=1.0/full_layers[-1]))
        b = tf.Variable(tf.random_normal([output_size], stddev=1.0/full_layers[-1]))
    else:
        index = len(weights)
        W = tf.Variable(saved_state[0][index])
        b = tf.Variable(saved_state[1][index])
    weights.append(W)
    biases.append(b)

    w_holder = [tf.placeholder(tf.float32, w.get_shape()) for w in weights]
    b_holder = [tf.placeholder(tf.float32, b.get_shape()) for b in  biases]
    w_assign = [w.assign(p) for w,p in zip(weights,w_holder)]
    b_assign = [b.assign(p) for b,p in zip( biases,b_holder)]

    y = tf.nn.softmax(tf.matmul(layers[-1], W) + b)

    cross_entropy = - tf.reduce_mean(tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=1))
    #train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.Session(
            config=tf.ConfigProto(
                inter_op_parallelism_threads=inter_threads,
                intra_op_parallelism_threads=intra_threads))
    
    correct = tf.nn.in_top_k(y, tf.argmax(y_, 1), top)
    accuracy = tf.reduce_sum(tf.cast(correct, tf.int32))

    sess.run(init)

    ops = {
            "sess":sess,
            "x":x,
            "y_":y_,
            "weights":weights,
            "biases":biases,
            "w_holder":w_holder,
            "b_holder":b_holder,
            "w_assign":w_assign,
            "b_assign":b_assign,
            "train_step":train_step,
            "cross_entropy":cross_entropy,
            "accuracy":accuracy,
            }

    return ops


def run_graph(
        data,
        labels,
        train_batch,
        ops,
        saved_state):


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
    weights = ops["weights"]
    biases = ops["biases"]
    w_holder = ops["w_holder"]
    b_holder = ops["b_holder"]
    w_assign = ops["w_assign"]
    b_assign = ops["b_assign"]
    train_step = ops["train_step"]
    cross_entropy =  ops["cross_entropy"]
    accuracy =  ops["accuracy"]

    # use saved state to assign saved weights and biases
    if saved_state is not None:
        feed_dict = {}
        for d,p in zip(saved_state[0],w_holder):
            feed_dict[p] = d
        for d,p in zip(saved_state[1],b_holder):
            feed_dict[p] = d
        sess.run(w_assign+b_assign, feed_dict=feed_dict)

    number_of_batches = len(data)/train_batch
    time_this = time.time()
    min_batches = comm.allreduce(number_of_batches, MPI.MIN)
    time_comm += time.time() - time_this
    if number_of_batches == 0:
        number_of_batches = 1
    
    for i in range(number_of_batches):
        lo = i*train_batch
        hi = (i+1)*train_batch
        batch_xs = data[lo:hi]
        batch_ys = labels[lo:hi]
        sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
        if (i < min_batches) and (merge_every>=1) and (i%merge_every == 0):
            time_this = time.time()
            r_weights = sess.run(weights)
            r_biases = sess.run(biases)
            for r in r_weights:
                comm.Allreduce(MPI.IN_PLACE, r, MPI.SUM)
                r /= size
            for r in r_biases:
                comm.Allreduce(MPI.IN_PLACE, r, MPI.SUM)
                r /= size
            feed_dict = {}
            for d,p in zip(r_weights,w_holder):
                feed_dict[p] = d
            for d,p in zip(r_biases,b_holder):
                feed_dict[p] = d
            sess.run(w_assign+b_assign, feed_dict=feed_dict)
            time_comm += time.time() - time_this

    # average as soon as we're done with all batches so the error and
    # accuracy reflect the current epoch
    time_this = time.time()
    r_weights = sess.run(weights)
    r_biases = sess.run(biases)
    for r in r_weights:
        comm.Allreduce(MPI.IN_PLACE, r, MPI.SUM)
        r /= size
    for r in r_biases:
        comm.Allreduce(MPI.IN_PLACE, r, MPI.SUM)
        r /= size
    time_comm += time.time() - time_this
    feed_dict = {}
    for d,p in zip(r_weights,w_holder):
        feed_dict[p] = d
    for d,p in zip(r_biases,b_holder):
        feed_dict[p] = d
    sess.run(w_assign+b_assign, feed_dict=feed_dict)
    time_comm += time.time() - time_this

    sum_error = 0.0
    if error_batch:
        for i in range(number_of_batches):
            lo = i*train_batch
            hi = (i+1)*train_batch
            batch_xs = data[lo:hi]
            batch_ys = labels[lo:hi]
            sum_error += sess.run(cross_entropy, feed_dict={x: batch_xs, y_:batch_ys})
    else:
        sum_error = sess.run(cross_entropy, feed_dict={x: data, y_: labels})
    time_this = time.time()
    sum_error_all = comm.allreduce(sum_error)
    time_comm += time.time() - time_this
    accurate = 0.0
    if error_batch:
        test_batch_count = len(test_dat)/train_batch
        if test_batch_count == 0:
            test_batch_count = 1
        for i in range(test_batch_count):
            lo = i*train_batch
            hi = (i+1)*train_batch
            batch_xs = test_dat[lo:hi]
            batch_ys = test_lab[lo:hi]
            accurate += sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    else:
        accurate = sess.run(accuracy, feed_dict={x: test_dat, y_: test_lab})
    time_this = time.time()
    accurate = comm.allreduce(accurate, MPI.SUM)
    acc_count = comm.allreduce(len(test_dat), MPI.SUM)
    accurate = float(accurate)/acc_count
    time_comm += time.time() - time_this

    if accurate > max_accuracy_encountered_value:
        max_accuracy_encountered_value = accurate
        max_accuracy_encountered_epoch = epoch
        max_accuracy_encountered_time = time.time()-time_global_start

    time_all = time.time() - time_epoch_start

    if 0 == rank:
        print "%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % (
                epoch+1,
                time_all,
                time.time()-time_global_start,
                accurate,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.0,
                time_comm / time_all,
                sum_error_all/size,
                )
    sys.stdout.flush()

    return r_weights, r_biases


if 0 == rank:
    print "epoch,etime,ctime,accuracy,MB_mem,time_comm,error"

data_threshold = int(len(full_dat)/2)
active_dat = full_dat
active_lab = full_lab
inactive_dat = np.empty([0]+list(full_dat.shape[1:]), full_dat.dtype)
inactive_lab = np.empty([0]+list(full_lab.shape[1:]), full_lab.dtype)

if stop_time > 0:
    saved_state = None
    ops = populate_graph(
            conv_layers,
            full_layers,
            learning_rate,
            input_shape,
            saved_state)
    while stop_time > (time.time()-time_global_start):
        saved_state = run_graph(
                active_dat,
                active_lab,
                train_batch,
                ops,
                saved_state)
        epoch += 1

else:
    saved_state = None
    ops = populate_graph(
            conv_layers,
            full_layers,
            learning_rate,
            input_shape,
            saved_state)
    for epoch in range(epochs):
        saved_state = run_graph(
                active_dat,
                active_lab,
                train_batch,
                ops,
                saved_state)

if 0 == rank:
    print "max accuracy achieved value", max_accuracy_encountered_value
    print "max accuracy achieved epoch", max_accuracy_encountered_epoch+1
    print "max accuracy achieved time", max_accuracy_encountered_time

