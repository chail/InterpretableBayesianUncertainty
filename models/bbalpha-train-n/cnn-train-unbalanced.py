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
from collections import Counter
from BBalpha_dropout import *

if len(sys.argv) != 4:
    print("Call this program like this:\n"
          "    ./cnn-train-unbalanced.py run frac targetclass\n"
          "    e.g. ./cnn-train-unbalanced.py 1 0.1 5\n"
          "Increase the fraction of examples in the targetclass"
          " class relative to all other classes. Frac specifies"
          "the fraction of target total used in training and validation"
          )
    exit()

# extract command line arguments
dataset = 'cifar10'
alpha = 0.5
run = sys.argv[1]
frac = float(sys.argv[2])
target = int(sys.argv[3])

train, validation, _ = load_dataset.load_image_data(dataset,
                                                    channels_first=False)

# extract target vs. other, will try to classify these 2 digits
train_target_ind = np.where(np.argmax(train[1], axis=1) == target)
train_other_ind = np.where(np.argmax(train[1], axis=1) != target)
train_other = (train[0][train_other_ind],
               train[1][train_other_ind])

validation_target_ind = np.where(np.argmax(validation[1], axis=1)
                                 == target)
validation_other_ind = np.where(np.argmax(validation[1], axis=1)
                                != target)
validation_other = (validation[0][validation_other_ind],
                    validation[1][validation_other_ind])

# take a fraction of 3's from train and val
n_target_total_train = len(train_target_ind[0])
n_target_frac_train = int(n_target_total_train * frac)
frac_ind_target_train = train_target_ind[0][:n_target_frac_train]
train_target_frac = (train[0][frac_ind_target_train],
                     train[1][frac_ind_target_train])

n_target_total_validation = len(validation_target_ind[0])
n_target_frac_validation = int(n_target_total_validation * frac)
frac_ind_target_validation = \
        validation_target_ind[0][:n_target_frac_validation]
validation_target_frac = (validation[0][frac_ind_target_validation],
                          validation[1][frac_ind_target_validation])

# train and val sets are all other classes and a fraction of target
train = (np.concatenate((train_other[0], train_target_frac[0]), axis=0),
         np.concatenate((train_other[1], train_target_frac[1]), axis=0))
validation = (np.concatenate((validation_other[0],
                              validation_target_frac[0]), axis=0),
              np.concatenate((validation_other[1],
                              validation_target_frac[1]), axis=0))

# permute data
n_train = n_target_frac_train + len(train_other_ind[0])
n_val = n_target_frac_validation + len(validation_other_ind[0])
perm_train = np.random.permutation(n_train)
perm_val = np.random.permutation(n_val)
train = (train[0][perm_train], train[1][perm_train])
validation = (validation[0][perm_val], validation[1][perm_val])

print("Training examples")
print(sorted(Counter(np.argmax(train[1], axis=1)).items()))
print("Validation examples")
print(sorted(Counter(np.argmax(validation[1], axis=1)).items()))
print("Training data shape: x:{} y:{}".format(train[0].shape,
                                              train[1].shape))
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
nb_classes = train[1].shape[1]

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
directory = os.path.join('saved_models_unbalanced',
                         '{}-cnn-alpha{}-run{}-target{}'
                         .format(dataset, alpha, run, target),
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
