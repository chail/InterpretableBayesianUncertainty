#!/usr/bin/env python3
# coding: utf-8

from toolbox import load_dataset

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.models import Model
from keras import optimizers
from keras.layers import Input
import numpy as np
import os
import pickle
import sys
from timeit import default_timer as timer
from collections import defaultdict
from BBalpha_dropout import *

if len(sys.argv) != 3:
    print("Call this program like this:\n"
          "    ./cnn-train-mnist.py run frac\n"
          "    e.g. ./cnn-train-mnist.py 1 0.1\n"
          "This task classifies 3's from 8's. Frac specifies"
          "the fraction of total 3's used in training and validation"
          )
    exit()

# extract command line arguments
dataset = 'mnist'
alpha = 0.5
run = sys.argv[1]
frac = float(sys.argv[2])

train, validation, _ = load_dataset.load_image_data(dataset,
                                                    channels_first=False)

# extract 3's and 8's, will try to classify these 2 digits
train_3_ind = np.where(np.argmax(train[1], axis=1) == 3)
train_8_ind = np.where(np.argmax(train[1], axis=1) == 8)
train_8 = (train[0][train_8_ind], train[1][train_8_ind][:, [3, 8]])

validation_3_ind = np.where(np.argmax(validation[1], axis=1) == 3)
validation_8_ind = np.where(np.argmax(validation[1], axis=1) == 8)
validation_8 = (validation[0][validation_8_ind],
                validation[1][validation_8_ind][:, [3, 8]])

# take a fraction of 3's from train and val
n_3_total_train = len(train_3_ind[0])
n_3_frac_train = int(n_3_total_train * frac)
frac_ind_3_train = train_3_ind[0][:n_3_frac_train]
train_3_frac = (train[0][frac_ind_3_train],
                train[1][frac_ind_3_train][:, [3, 8]])

n_3_total_validation = len(validation_3_ind[0])
n_3_frac_validation = int(n_3_total_validation * frac)
frac_ind_3_validation = validation_3_ind[0][:n_3_frac_validation]
validation_3_frac = (validation[0][frac_ind_3_validation],
                     validation[1][frac_ind_3_validation][:, [3, 8]])

# train and val sets are all 8's and a fraction of 3's
train = (np.concatenate((train_8[0], train_3_frac[0]), axis=0),
         np.concatenate((train_8[1], train_3_frac[1]), axis=0))
validation = (np.concatenate((validation_8[0], validation_3_frac[0]), axis=0),
              np.concatenate((validation_8[1], validation_3_frac[1]), axis=0))

# permute data
n_train = n_3_frac_train + len(train_8_ind[0])
n_val = n_3_frac_validation + len(validation_8_ind[0])
perm_train = np.random.permutation(n_train)
perm_val = np.random.permutation(n_val)
train = (train[0][perm_train], train[1][perm_train])
validation = (validation[0][perm_val], validation[1][perm_val])

print("Number of 8's: in training: {} in validation: {}"
      .format(train_8[1].shape[0], validation_8[1].shape[0]))
print("Number of 3's: in training: {} in validation: {}"
      .format(train_3_frac[1].shape[0], validation_3_frac[1].shape[0]))
print("Training data shape: x:{} y:{}".format(train[0].shape, train[1].shape))
print("Val data shape: x:{} y:{}".format(validation[0].shape,
                                         validation[1].shape))


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
nb_classes = 2  # just 2 classes, 3 and 8

batch_size = 128
nb_layers = 2
nb_units = 100
p = 0.5
wd = 1e-6

K_mc = 10

epochs = 5

# model layers
assert K.image_data_format() == 'channels_last', \
        'use a backend with channels last'
input_shape = (input_dim[0], input_dim[1], input_channels)  # (dimX, )
inp = Input(shape=input_shape)
layers = get_logit_cnn_layers(nb_units, p, wd, nb_classes, layers=[])

# build model with MC samples
mc_logits = GenerateMCSamples(inp, layers, K_mc)  # repeats stochastic layers K_mc times
# if alpha = 0.0, bbalpha returns categorical cross entropy with logits
loss_function = bbalpha_softmax_cross_entropy_with_mc_logits(alpha)
model = Model(inputs=inp, outputs=mc_logits)
opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=opt, loss=loss_function,
              metrics=['accuracy', loss_function, metric_avg_acc, metric_avg_ll])


train_Y_dup = np.squeeze(np.concatenate(K_mc * [train[1][:, None]], axis=1)) # N x K_mc x D
val_Y_dup = np.squeeze(np.concatenate(K_mc * [validation[1][:, None]], axis=1)) # N x K_mc x D

# training loop
directory = os.path.join('saved_models_mnist',
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
                        initial_epoch = ep, epochs=ep+1,
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
