from __future__ import print_function
import tensorflow as tf
import numpy as np
from keras.metrics import categorical_crossentropy, categorical_accuracy, top_k_categorical_accuracy
import time
from keras import backend as K
import keras_helpers

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--train_batch', type=int, default=256)
parser.add_argument('--iterations', type=int, default=500)
parser.add_argument('--network', type=str, default="AlexNet")
args = parser.parse_args()

if args.network == "AlexNet":
    net = getattr(keras_helpers, args.network)()
    data_shape = [227, 227, 3]
elif args.network == "InceptionV3":
    net = getattr(keras_helpers, args.network)()
    data_shape = [224, 224, 3]
elif args.network == "ResNet50":
    net = getattr(keras_helpers, args.network)()
    data_shape = [224, 224, 3]
elif args.network == "GoogLeNet":
    net = getattr(keras_helpers, args.network)()
    data_shape = [224, 224, 3]
else:
    sys.exit("Unknown Network")

fake_data = np.random.rand(args.train_batch, data_shape[0], data_shape[1], data_shape[2])
tmp_fake_labels = np.random.randint(0, high=1000, size=args.train_batch)
fake_labels = np.zeros([args.train_batch, 1000])
for i in range(args.train_batch):
    fake_labels[i, tmp_fake_labels[i]] = 1

loss = categorical_crossentropy(net.y_, net.y)
top1 = categorical_accuracy(net.y_, net.y)
top5 = top_k_categorical_accuracy(net.y_, net.y, 5)

base_lr = 0.02
step = tf.Variable(0, trainable=False, name="Step")
learning_rate = tf.train.exponential_decay(base_lr, step, 1, 0.999964)

weight_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name[-3:] == "W:0"]
bias_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name[-3:] == "b:0"]

optimizer1 = tf.train.MomentumOptimizer(learning_rate, 0.9)
optimizer2 = tf.train.MomentumOptimizer(tf.scalar_mul(2.0, learning_rate), 0.9)
grads = optimizer1.compute_gradients(loss, var_list=weight_list+bias_list)
w_grads = grads[:len(weight_list)]
b_grads = grads[len(weight_list):]

train1 = optimizer1.apply_gradients(w_grads, global_step=step)
train2 = optimizer2.apply_gradients(b_grads, global_step=step)
train_step = tf.group(train1, train2)

init = tf.global_variables_initializer()
sess = tf.Session()
K.set_session(sess)

sess.run(init)

start = time.time()
total_it = 0
while total_it < args.iterations:
    batch_start = time.time()
    feeder = dict()
    feeder[net.x] = fake_data
    feeder[net.y_] = fake_labels
    feeder[K.learning_phase()] = 1
    sess.run(train_step, feed_dict=feeder)
    total_it += 1
print("Iterations Per Second", args.iterations/(time.time() - start), "for", args.network)
