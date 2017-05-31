from __future__ import print_function, division
import numpy as np
from keras.models import Model
from keras.optimizers import SGD
import keras_helpers
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--train_batch', type=int, default=256)
parser.add_argument('--iterations', type=int, default=500)
parser.add_argument('--batches_per_record', type=int, default=20)
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

model = Model(input=net.x, output=net.y)
model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=['categorical_crossentropy'],
              metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])
print(model.summary())
model.fit(fake_data, fake_labels,
          validation_data=(fake_data, fake_labels),
          nb_epoch=args.iterations//args.batches_per_record, batch_size=args.train_batch, verbose=1)
