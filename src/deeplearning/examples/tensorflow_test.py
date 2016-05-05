import tensorflow as tf
import time
import resource
from mnist_mpi_reader import read_data_sets

mnist = read_data_sets('MNIST_data', one_hot=True, validation_percentage=1.0 / 6)
train_data = mnist.train.images
train_labels = mnist.train.labels
validation_data = mnist.validation.images
validation_labels = mnist.validation.labels
test_data = mnist.test.images
test_labels = mnist.test.labels
shape = [28, 28, 1]
size = 28 * 28 * 1
classes = 10
top = 2

# set up network

learning_rate = 0.1
train_batch = 10

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_shaped = tf.reshape(x, [-1, 28, 28, 1])
W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 20], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[20]))
conv1 = tf.nn.conv2d(x_shaped, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
relu1 = tf.nn.relu(pool1)
res = tf.reshape(relu1, [-1, 3920])
W2 = tf.Variable(tf.random_normal([3920, 10], stddev=1.0/3920))
b2 = tf.Variable(tf.random_normal([10], stddev=1.0/3920))
y = tf.nn.softmax(tf.matmul(res, W2)+b2)

cross_entropy = - tf.reduce_mean(tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=1))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()

correct1 = tf.nn.in_top_k(y, tf.argmax(y_, 1), 1)
accuracy1 = tf.reduce_sum(tf.cast(correct1, tf.int32))
sess.run(init)

print "epoch,etime,ctime,accuracy,top_k,MB_mem,error"

time_global_start = time.time()
for epoch in range(30):
    time_epoch_start = time.time()
    number_of_batches = len(train_data) / train_batch
    for i in range(number_of_batches):
        lo = i * train_batch
        hi = (i + 1) * train_batch
        batch_xs = train_data[lo:hi]
        batch_ys = train_labels[lo:hi]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    acc1 = 0.0
    for i in range(100):
        lo = i * 100
        hi = (i + 1) * 100
        batch_xs = test_data[lo:hi]
        batch_ys = test_labels[lo:hi]
        acc1 += sess.run(accuracy1, feed_dict={x: batch_xs, y_: batch_ys})
    acc1 /= float(len(test_data))

    print "%s,%s,%s,%s,%s" % (
        epoch + 1,
        time.time() - time_epoch_start,
        time.time() - time_global_start,
        acc1,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.0
    )

print "Average epoch:", (time.time()-time_global_start)/30
