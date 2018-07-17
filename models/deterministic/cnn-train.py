#!/usr/bin/env python3
# coding: utf-8

from toolbox import load_dataset

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.callbacks import Callback
from keras import optimizers
from keras.utils import np_utils
import numpy as np
import os, pickle, sys
from timeit import default_timer as timer
from collections import defaultdict
from keras.layers import Input
from models import *

if len(sys.argv) != 3:
    print("Call this program like this:\n"
          "    ./cnn-train.py dataset run\n"
          "    e.g. ./cnn-train.py mnist 1\n"
          "Dataset is either ['mnist', 'cifar10', 'svhn']"
         )
    exit()

# extract command line arguments
dataset = sys.argv[1]
run = sys.argv[2]

# get dataset
if dataset == 'mnist':
    train, validation, _ = load_dataset.load_mnist(flatten=False,
                                                   channels_first=False)
elif dataset == 'cifar10':
    train, validation, _ = load_dataset.load_cifar10(channels_first=False)
elif dataset == 'svhn':
    train, validation, _ = load_dataset.load_svhn(channels_first=False)
else:
    print("Unrecognized dataset, use 'mnist', 'cifar10', or 'svhn'")
    print("Exiting...")
    exit()

# otherwise TF grabs all available gpu memory
if not hasattr(K, "tf"):
    raise RuntimeError("This code requires keras to be configured"
                       " to use the TensorFlow backend.")

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

# constants
nb_train = train[0].shape[0]
nb_val = validation[0].shape[0]
input_dim = (train[0].shape[1], train[0].shape[2])
input_channels = train[0].shape[3]
nb_classes = train[1].shape[1]

batch_size = 128
nb_layers = 2
nb_units = 100
p = 0.5
wd = 1e-6

epochs = 30

# model layers
assert K.image_data_format() == 'channels_last', \
        'use a backend with channels last'
input_shape = (input_dim[0], input_dim[1], input_channels) # (dimX, )
inp = Input(shape=input_shape)

# cnn model
model =  build_cnn(inp, p, nb_units, nb_classes, wd)
opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy', 'categorical_crossentropy'])


# training loop
directory = os.path.join('saved_models',
                         '{}-cnn-run{}'.format(dataset, run))
os.makedirs(directory, exist_ok=True)





results = defaultdict(list)
min_val = float('inf')
min_val_ep = 0
ep = 0

while ep < max(2 * min_val_ep, epochs):
    tic = timer()
    history = model.fit(train[0], train[1],
                        verbose=1, batch_size=batch_size,
                        initial_epoch = ep, epochs=ep+1,
                        validation_data=(validation[0], validation[1]))
    toc = timer()
    results['train_N'].append(train[0].shape[0])
    results['val_N'].append(validation[0].shape[0])
    results['time'].append(toc-tic)
    results['train_total_loss'].extend(history.history['loss'])
    results['train_categorical_crossentropy'].extend(history.history['categorical_crossentropy'])
    results['train_acc'].extend(history.history['acc'])
    results['val_total_loss'].extend(history.history['val_loss'])
    results['val_categorical_crossentropy'].extend(history.history['val_categorical_crossentropy'])
    results['val_acc'].extend(history.history['val_acc'])

    val_categorical_crossentropy = results['val_categorical_crossentropy'][-1]
    if val_categorical_crossentropy < min_val:
        min_val = val_categorical_crossentropy
        min_val_ep = ep
        print("Updating min_val_ep: {}".format(min_val_ep))

        # save the model
        tic = timer()
        model.save(os.path.join(directory, 'model.h5'))
        toc = timer()
    print("Min_val_ep: {}\t Min_val: {:.3f}".format(min_val_ep, min_val))
    ep += 1

    # save result after every epoch
    with open(os.path.join(directory, 'results.p'), 'wb') as f:
        pickle.dump(results, f)

