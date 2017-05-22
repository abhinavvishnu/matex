from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_batch', type=int, default=64, help='training batch size')
args = parser.parse_args()

mnist = tf.DataSet("MNIST", normalize=255.0)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])

w1 = tf.Variable(tf.random_uniform([5, 5, 1, 20], minval=-np.sqrt(6.0/21), maxval=np.sqrt(6.0/21), dtype=tf.float32))
b1 = tf.Variable(tf.random_uniform([20], minval=-np.sqrt(6.0/21), maxval=np.sqrt(6.0/21), dtype=tf.float32))

conv_1 = tf.nn.bias_add(tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME'), b1)
pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
relu1 = tf.nn.relu(pool_1)

w2 = tf.Variable(tf.random_uniform([5, 5, 20, 50], minval=-np.sqrt(6.0/70), maxval=np.sqrt(6.0/70), dtype=tf.float32))
b2 = tf.Variable(tf.random_uniform([50], minval=-np.sqrt(6.0/70), maxval=np.sqrt(6.0/70), dtype=tf.float32))

conv_2 = tf.nn.bias_add(tf.nn.conv2d(relu1, w2, strides=[1, 1, 1, 1], padding='SAME'), b2)
pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
relu2 = tf.nn.relu(pool_2)

w3 = tf.Variable(tf.random_uniform([2450, 500], minval=-np.sqrt(6.0/2950), maxval=np.sqrt(6.0/2950), dtype=tf.float32))
b3 = tf.Variable(tf.random_uniform([500], minval=-np.sqrt(6.0/2950), maxval=np.sqrt(6.0/2950), dtype=tf.float32))

flat_pool = tf.reshape(relu2, [-1, 2450])
fc3 = tf.nn.bias_add(tf.matmul(flat_pool, w3), b3)
relu3 = tf.nn.relu(fc3)

w4 = tf.Variable(tf.random_uniform([500, 10], minval=-np.sqrt(6.0/510), maxval=np.sqrt(6.0/510), dtype=tf.float32))
b4 = tf.Variable(tf.random_uniform([10], minval=-np.sqrt(6.0/510), maxval=np.sqrt(6.0/510), dtype=tf.float32))

y = tf.nn.softmax(tf.nn.bias_add(tf.matmul(relu3, w4), b4))
correct = tf.nn.in_top_k(y_, tf.argmax(y, 1), 1)
accuracy = tf.reduce_sum(tf.cast(correct, tf.int32))

cross_entropy = - tf.reduce_mean(tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=1))
weight_decay = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4)
bias_decay = tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(b3) + tf.nn.l2_loss(b4)
decay_coefficient = 0.0001
cross_entropy += decay_coefficient*weight_decay + decay_coefficient*bias_decay

step = tf.Variable(0, trainable=False, name="Step")
learning_rate = 0.01
learning_rate = tf.train.inverse_time_decay(learning_rate, step, 1, 0.0001)
opt1 = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_step = opt1.minimize(cross_entropy, global_step=step)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_it = 0
global_start = time.time()
while total_it < 10000:
    epoch_start = time.time()
    for train_batch in range(int(len(mnist.training_data)/args.train_batch)):
        batch_start = time.time()

        lo = train_batch * args.train_batch
        hi = (train_batch+1)*args.train_batch
        batch_x = mnist.training_data[lo:hi]
        batch_y = mnist.training_labels[lo:hi]
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
        sess.run(learning_rate)
        total_it += 1
        if total_it % 100 == 0:
            batch_err = sess.run(tf.reduce_mean(cross_entropy), feed_dict={x: batch_x, y_: batch_y})
            print("Batch Loss for", total_it, "is", batch_err, "in time", time.time() - batch_start)
        if total_it % 500 == 0:
            test_start = time.time()
            acc = 0.0
            test_loss = 0.0
            for test_batch in range(int(len(mnist.testing_data)/100)):
                lo = test_batch * 100
                hi = (test_batch + 1) * 100
                batch_x = mnist.testing_data[lo:hi]
                batch_y = mnist.testing_labels[lo:hi]
                acc += sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y})
                test_loss += sess.run(tf.reduce_mean(cross_entropy), feed_dict={x: batch_x, y_: batch_y})
            acc /= len(mnist.testing_data)
            test_loss /= 100.0
            print("Test Loss", total_it, "is", test_loss, "in time", time.time() - test_start)
            print("Test Accuracy", total_it, "is", acc, "in time", time.time() - test_start)
print("Iterations per second", total_it/(time.time() - global_start))
