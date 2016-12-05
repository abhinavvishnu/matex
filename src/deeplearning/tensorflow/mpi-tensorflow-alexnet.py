import tensorflow as tf
from pnetcdf import read_pnetcdf
import numpy as np
import time
import argparse
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, default=None, help='filename for CSV or PNETCDF')
parser.add_argument('--test_data', type=str, default=None, help='filename for CSV or PNETCDF')
args = parser.parse_args()

average = np.load('ilsvrc_2012_mean.npy')
average = np.transpose(average, [1, 2, 0])

data_start = time.time()

training_data, training_labels = read_pnetcdf(args.train_data)
testing_data, testing_labels = read_pnetcdf(args.test_data)

print "Time to Load Data", time.time()-data_start

l2_coeff = 0.0005
train_batch_size = 256
test_batch_size = 100


def mpi_average(tensor):
    comm.Allreduce(MPI.IN_PLACE, tensor, MPI.SUM)
    tensor /= size
    return tensor


def create_weights(shape, std):
    initial_weights = tf.random_normal(shape, stddev=std)
    return tf.Variable(initial_weights)


def create_biases(shape, val):
    initial_biases = tf.constant(val, shape=shape)
    return tf.Variable(initial_biases)

data = tf.placeholder(tf.float32, [None, 224, 224, 3])
labels = tf.placeholder(tf.float32, [None, 1000])
keep_prob = tf.placeholder(tf.float32)


class Layer:
    def __init__(self):
        self.layer = None
        self.weights = None
        self.biases = None


def create_conv(prev_layer, window_size, stride, features_in, features_out, w_std, bias_val, groups=1):
    output = Layer()
    output.weights = create_weights([window_size, window_size, features_in, features_out], w_std)
    output.biases = create_biases([features_out], bias_val)
    if groups != 1:
        prev_groups = tf.split(3, groups, prev_layer)
        weight_groups = tf.split(3, groups, output.weights)
        bias_groups = tf.split(0, groups, output.biases)
        output_groups = [tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(i, w, strides=[1, stride, stride, 1], padding='SAME'), b)) for i, w, b in
                         zip(prev_groups, weight_groups, bias_groups)]
        t_output = tf.concat(3, output_groups)
        output.layer = tf.reshape(t_output, [-1]+t_output.get_shape().as_list()[1:])
    else:
        output.layer = tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(prev_layer, output.weights, strides=[1, stride, stride, 1], padding='SAME'), output.biases))
    return output


def create_pool(prev_layer, window_size, stride):
    return tf.nn.max_pool(prev_layer, ksize=[1, window_size, window_size, 1],
                          strides=[1, stride, stride, 1], padding='SAME')


def create_lrn(prev_layer, depth, bias, alpha, beta):
    return tf.nn.local_response_normalization(prev_layer, depth_radius=depth, bias=bias, alpha=alpha, beta=beta)


def create_full(prev_layer, features_in, features_out, w_std, bias_val):
    output = Layer()
    output.weights = create_weights([features_in, features_out], w_std)
    output.biases = create_biases([features_out], bias_val)
    output.layer = tf.nn.relu(tf.matmul(prev_layer, output.weights) + output.biases)
    return output


def create_dropout(prev_layer, prob_placeholder):
    return tf.nn.dropout(prev_layer, prob_placeholder)

cost = 0.0

conv_1 = create_conv(data, 11, 4, 3, 96, 0.01, 0.0)
lrn_1 = create_lrn(conv_1.layer, 5, 1, 0.0001, 0.75)
pool_1 = create_pool(lrn_1, 3, 2)
conv_2 = create_conv(pool_1, 5, 1, 96/2, 256, 0.01, 0.1, groups=2)
lrn_2 = create_lrn(conv_2.layer, 5, 1, 0.0001, 0.75)
pool_2 = create_pool(lrn_2, 3, 2)
conv_3 = create_conv(pool_2, 3, 1, 256, 384, 0.01, 0.0)
conv_4 = create_conv(conv_3.layer, 3, 1, 384/2, 384, 0.01, 0.1, groups=2)
conv_5 = create_conv(conv_4.layer, 3, 1, 384/2, 256, 0.01, 0.1, groups=2)
pool_3 = create_pool(conv_5.layer, 3, 2)
flattened = tf.reshape(pool_3, [-1, 12544])
fc6 = create_full(flattened, 12544, 4096, 0.005, 0.1)
drop_6 = create_dropout(fc6.layer, keep_prob)
fc7 = create_full(drop_6, 4096, 4096, 0.005, 0.1)
drop_7 = create_dropout(fc7.layer, keep_prob)
score = create_full(drop_7, 4096, 1000, 0.01, 0.0)
softmax = tf.nn.softmax(score.layer)
loss = tf.nn.softmax_cross_entropy_with_logits(score.layer, labels)

cost += l2_coeff * tf.nn.l2_loss(conv_1.weights)
cost += l2_coeff * tf.nn.l2_loss(conv_2.weights)
cost += l2_coeff * tf.nn.l2_loss(conv_3.weights)
cost += l2_coeff * tf.nn.l2_loss(conv_4.weights)
cost += l2_coeff * tf.nn.l2_loss(conv_5.weights)
cost += l2_coeff * tf.nn.l2_loss(fc6.weights)
cost += l2_coeff * tf.nn.l2_loss(fc7.weights)
cost += l2_coeff * tf.nn.l2_loss(score.weights)
cost += tf.reduce_mean(loss)

base_lr = 0.01
learning_rate = tf.Variable(base_lr)

weight_list = [conv_1.weights, conv_2.weights, conv_3.weights, conv_4.weights, conv_5.weights, fc6.weights,
               fc7.weights, score.weights]
bias_list = [conv_1.biases, conv_2.biases, conv_3.biases, conv_4.biases, conv_5.biases, fc6.biases,
             fc7.biases, score.biases]
opt1 = tf.train.MomentumOptimizer(learning_rate, 0.9)
opt2 = tf.train.MomentumOptimizer(tf.mul(2.0, learning_rate), 0.9)

grads = tf.gradients(cost, weight_list + bias_list)
grads1 = grads[:len(weight_list)]
grads2 = grads[len(weight_list):]

grads1 = [tf.py_func(mpi_average, [x], tf.float32) for x in grads1]
grads2 = [tf.py_func(mpi_average, [x], tf.float32) for x in grads2]

train1 = opt1.apply_gradients(zip(grads1, weight_list))
train2 = opt2.apply_gradients(zip(grads2, bias_list))
train_step = tf.group(train1, train2)

# optimizer = tf.train.GradientDescentOptimizer(args_in.learning_rate)
# grads_and_vars = optimizer.compute_gradients(cross_entropy, weights + biases)
# grads_and_vars = [(tf.py_func(mpi_average, [gv[0]], tf.float32), gv[1]) for gv in grads_and_vars]
# train_step = optimizer.apply_gradients(grads_and_vars)

init = tf.initialize_all_variables()
sess = tf.Session(config=tf.ConfigProto())

correct5 = tf.nn.in_top_k(labels, tf.argmax(softmax, 1), 5)
accuracy5 = tf.reduce_sum(tf.cast(correct5, tf.int32))

correct1 = tf.nn.in_top_k(labels, tf.argmax(softmax, 1), 1)
accuracy1 = tf.reduce_sum(tf.cast(correct1, tf.int32))

sess.run(init)

epochs = 90
start = time.time()
total_it = 0
epoch = 0

while total_it <= 360000:
    estart = time.time()
    for train_batch in range(len(training_data)/train_batch_size):
        bstart = time.time()
        x_offset = np.random.randint(0, high=(256-224))
        y_offset = np.random.randint(0, high=(256-224))
        x_range = range(x_offset, x_offset+224)
        y_range = range(y_offset, y_offset+224)
        temp_ave = average[np.ix_(x_range, y_range, range(3))]
        lo = train_batch * train_batch_size
        hi = (train_batch+1) * train_batch_size
        batch_x = training_data[np.ix_(range(lo, hi), range(3), x_range, y_range)]
        batch_x = np.transpose(batch_x, [0, 2, 3, 1])
        batch_x = batch_x.astype('float64')
        batch_x -= temp_ave
        temp_batch_y = training_labels[lo:hi]
        batch_y = np.zeros([hi-lo, 1000])
        for i in range(hi-lo):
            batch_y[i, temp_batch_y[i]] = 1
        drop_prob = 0.5
        feeder = dict()
        feeder[data] = batch_x
        feeder[labels] = batch_y
        feeder[keep_prob] = drop_prob
        sess.run(train_step, feed_dict=feeder)
        if total_it % 100000 == 0:
            learning_rate = tf.div(learning_rate, 10.0)
        if train_batch % 20 == 0:
            batch_err = sess.run(tf.reduce_mean(loss), feed_dict=feeder)
            print "Error for batch", total_it, "is", batch_err, "in time", time.time()-bstart
        total_it += 1
    acc1 = 0
    acc5 = 0
    for test_batch in range(len(testing_data)/test_batch_size):
        offset = (256-224)/2
        central_range = range(offset, offset+224)
        temp_ave = average[np.ix_(central_range, central_range, range(3))]
        lo = test_batch * test_batch_size
        hi = (test_batch + 1) * test_batch_size
        batch_x = testing_data[np.ix_(range(lo, hi), range(3), central_range, central_range)]
        batch_x = batch_x.astype('float64')
        batch_x = np.transpose(batch_x, [0, 2, 3, 1])
        batch_x -= temp_ave
        temp_batch_y = training_labels[lo:hi]
        batch_y = np.zeros([hi-lo, 1000])
        for i in range(hi-lo):
            batch_y[i, temp_batch_y[i]] = 1
        drop_prob = 1.0
        feeder = dict()
        feeder[data] = batch_x
        feeder[labels] = batch_y
        feeder[keep_prob] = drop_prob
        acc1 += sess.run(accuracy1, feed_dict=feeder)
        acc5 += sess.run(accuracy5, feed_dict=feeder)
    acc1 /= len(testing_data)
    acc5 /= len(testing_data)
    epoch += 1
    print epoch, acc1, acc5, time.time()-estart, time.time()-start
