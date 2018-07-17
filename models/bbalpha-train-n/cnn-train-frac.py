#!/usr/bin/env python3
# coding: utf-8

from toolbox import load_dataset

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.models import Model
from keras import optimizers
import numpy as np
import os
import pickle
import sys
from timeit import default_timer as timer
from collections import defaultdict
from collections import Counter
from BBalpha_dropout import *


if len(sys.argv) != 5:
    print("Call this program like this:\n"
          "    ./cnn-train-frac.py dataset alpha run frac\n"
          "    e.g. ./cnn-train-frac.py mnist 0.5 1 0.1\n"
          "Dataset is either ['mnist', 'cifar10', 'svhn']\n"
          "Frac is the fraction of training data to train on")
    exit()

# extract command line arguments
dataset = sys.argv[1]
alpha = float(sys.argv[2])
run = sys.argv[3]
frac = float(sys.argv[4])

train, validation, _ = load_dataset.load_image_data(dataset,
                                                    channels_first=False)

# take a fraction of the training data
n = train[0].shape[0]
n_frac = int(n * frac)
perm_ind = np.random.permutation(n)
train_perm = (train[0][perm_ind], train[1][perm_ind])
train = (train_perm[0][:n_frac], train_perm[1][:n_frac])

print("Taking {} of training data...".format(frac))
print("Items in each class:")
labels = list(np.argmax(train[1], axis=1))
label_count = Counter(labels)
[print("Class {}: {}".format(key, label_count[key])) for key in label_count]


# otherwise TF grabs all available gpu memory
if not hasattr(K, "tf"):
    raise RuntimeError("This code requires keras to be configured"
                       " to use the TensorFlow backend.")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
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

K_mc = 10

epochs = 30

# model layers
assert K.image_data_format() == 'channels_last', \
        'use a backend with channels last'
input_shape = (input_dim[0], input_dim[1], input_channels)  # (dimX, )
inp = Input(shape=input_shape)
layers = get_logit_cnn_layers(nb_units, p, wd, nb_classes, layers=[])

# build model with MC samples
mc_logits = GenerateMCSamples(inp, layers, K_mc) # repeats stochastic layers K_mc times
# if alpha = 0.0, bbalpha returns categorical cross entropy with logits
loss_function = bbalpha_softmax_cross_entropy_with_mc_logits(alpha)
model = Model(inputs=inp, outputs=mc_logits)
opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=opt, loss=loss_function,
              metrics=['accuracy', loss_function, metric_avg_acc, metric_avg_ll])


train_Y_dup = np.squeeze(np.concatenate(K_mc * [train[1][:, None]], axis=1)) # N x K_mc x D
val_Y_dup = np.squeeze(np.concatenate(K_mc * [validation[1][:, None]], axis=1)) # N x K_mc x D

# training loop
directory = os.path.join('saved_models_train_frac',
                         '{}-cnn-alpha{}-run{}'.format(dataset, alpha, run),
                         'frac{:.1f}'.format(frac))
os.makedirs(directory, exist_ok=True)
results = defaultdict(list)
min_val = float('inf')
min_val_ep = 0
ep = 0

while ep < max(2 * min_val_ep, epochs):
    tic = timer()
    history = model.fit(train[0], train_Y_dup,
                        verbose=1, batch_size=batch_size,
                        initial_epoch=ep, epochs=ep+1,
                        validation_data=(validation[0], val_Y_dup))
    toc = timer()
    results['train_N'].append(train[0].shape[0])
    results['val_N'].append(validation[0].shape[0])
    results['time'].append(toc-tic)
    results['train_total_loss'].extend(history.history['loss'])
    results['train_bbalpha_loss'].extend(history.history['bbalpha_loss'])
    results['train_acc'].extend(history.history['acc'])
    results['train_avg_acc'].extend(history.history['metric_avg_acc'])
    results['train_ll'].extend(history.history['metric_avg_ll'])
    results['val_total_loss'].extend(history.history['val_loss'])
    results['val_bbalpha_loss'].extend(history.history['val_bbalpha_loss'])
    results['val_acc'].extend(history.history['val_acc'])
    results['val_avg_acc'].extend(history.history['val_metric_avg_acc'])
    results['val_ll'].extend(history.history['val_metric_avg_ll'])

    val_bbalpha_loss = results['val_bbalpha_loss'][-1]
    if val_bbalpha_loss < min_val:
        min_val = val_bbalpha_loss
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


# load the last saved model and add uncertainties in a tf graph
filepath = os.path.join(directory, 'model.h5')
K_mc_test = 100
build_test_model(filepath, K_mc_test, p)
