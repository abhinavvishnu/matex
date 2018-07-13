import time
import tensorflow as tf

Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
l2 = tf.keras.regularizers.l2
Model = tf.keras.models.Model
Input = tf.keras.layers.Input
SGD = tf.keras.optimizers.SGD

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_batch', type=int, default=64, help='training batch size')
parser.add_argument('--epochs', type=int, default=26, help='Epochs')
args = parser.parse_args()

decay = 0.0001

start_time0 = time.time()
mnist = tf.DataSet("MNIST", normalize=255.0)

X_train = mnist.training_data
X_test = mnist.testing_data
y_train = mnist.training_labels
y_test = mnist.testing_labels
num_classes = y_test.shape[1]
print(num_classes)
start_time1 = time.time();
input_img = Input(shape=(28, 28, 1))
y = Conv2D(20, 5, kernel_regularizer=l2(decay), bias_regularizer=l2(decay),
                  padding='SAME', activation='relu')(input_img)
y = MaxPooling2D(strides=(2, 2), padding='SAME')(y)
y = Conv2D(50, 5, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding='SAME', activation='relu')(y)
y = MaxPooling2D(strides=(2, 2), padding='SAME')(y)
y = Flatten()(y)
y = Dense(500, activation='relu', kernel_regularizer=l2(decay), bias_regularizer=l2(decay))(y)
predict = Dense(10, activation='softmax', kernel_regularizer=l2(decay), bias_regularizer=l2(decay))(y)
model = Model(input_img, predict)
model.compile('SGD', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.epochs, batch_size=args.train_batch, verbose=2)
end_time = time.time()
print("Total Elapsed Time(NN): {}".format(end_time - start_time1))
print("Total Elapsed Time: {}".format(end_time - start_time0))
