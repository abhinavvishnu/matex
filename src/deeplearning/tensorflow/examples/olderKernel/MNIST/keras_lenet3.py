from __future__ import print_function
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD
from tensorflow import DataSet


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_batch', type=int, default=64)
parser.add_argument('--epochs', type=int, default=13)
args = parser.parse_args()


decay = 0.0001

mnist = DataSet("MNIST", normalize=255.0)

X_train = mnist.training_data
X_test = mnist.testing_data
y_train = mnist.training_labels
y_test = mnist.testing_labels
num_classes = y_test.shape[1]
print(num_classes)
input_img = Input(shape=(28, 28, 1))
y = Convolution2D(20, 5, 5, W_regularizer=l2(decay), b_regularizer=l2(decay),
                  border_mode='same', activation='relu')(input_img)
y = MaxPooling2D(strides=(2, 2), border_mode='same')(y)
y = Convolution2D(50, 5, 5, W_regularizer=l2(decay), b_regularizer=l2(decay), border_mode='same', activation='relu')(y)
y = MaxPooling2D(strides=(2, 2), border_mode='same')(y)
y = Flatten()(y)
y = Dense(500, activation='relu', W_regularizer=l2(decay), b_regularizer=l2(decay))(y)
predict = Dense(10, activation='softmax', W_regularizer=l2(decay), b_regularizer=l2(decay))(y)
model = Model(input=input_img, output=predict)
model.compile(optimizer=SGD(lr=0.01, decay=1e-4), loss=['categorical_crossentropy'], metrics=['categorical_accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=args.epochs, batch_size=args.batch_size, verbose=2)
