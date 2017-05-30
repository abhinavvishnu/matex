from __future__ import print_function
import tensorflow as tf
import time
import argparse
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Input
from keras.regularizers import l2
from keras.metrics import categorical_crossentropy

parser = argparse.ArgumentParser()
parser.add_argument('--train_batch', type=int, default=64)
parser.add_argument('--decay_coefficient', type=float, default=0.0001)
parser.add_argument('--valid_pct', type=float, default=1.0/6)
parser.add_argument('--data', type=str, default='MNIST')
parser.add_argument('--iterations', type=int, default='10000')
args = parser.parse_args()

mnist = tf.DataSet("MNIST", normalize=255.0)

x = Input(shape=(28, 28, 1))
y_ = tf.placeholder(tf.float32, [None, 10])
y = Convolution2D(20, 5, 5, W_regularizer=l2(args.decay_coefficient), b_regularizer=l2(args.decay_coefficient),
                  border_mode='same', activation='relu')(x)
y = MaxPooling2D(strides=(2, 2), border_mode='same')(y)
y = Convolution2D(50, 5, 5, W_regularizer=l2(args.decay_coefficient), b_regularizer=l2(args.decay_coefficient),
                  border_mode='same', activation='relu')(y)
y = MaxPooling2D(strides=(2, 2), border_mode='same')(y)
y = Flatten()(y)
y = Dense(500, activation='relu', W_regularizer=l2(args.decay_coefficient), b_regularizer=l2(args.decay_coefficient))(y)
y = Dense(10, activation='softmax', W_regularizer=l2(args.decay_coefficient),
          b_regularizer=l2(args.decay_coefficient))(y)
correct = tf.nn.in_top_k(y_, tf.argmax(y, 1), 1)
accuracy = tf.reduce_sum(tf.cast(correct, tf.int32))

cross_entropy = categorical_crossentropy(y_, y)

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
while total_it < args.iterations:
    epoch_start = time.time()
    for train_batch in range(int(len(mnist.training_data)/args.train_batch)):
        batch_start = time.time()
        lo = train_batch * args.train_batch
        hi = (train_batch+1)*args.train_batch
        batch_x = mnist.training_data[lo:hi]
        batch_y = mnist.training_labels[lo:hi]
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
        total_it += 1
        if total_it % 100 == 0:
            batch_err = sess.run(tf.reduce_mean(cross_entropy), feed_dict={x: batch_x, y_: batch_y})
            print("Batch Loss for", total_it, "is", batch_err, "in time", time.time() - batch_start)
        if total_it % 500 == 0:
            test_start = time.time()
            acc = 0.0
            test_loss = 0.0
            for test_batch in range(100):
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
