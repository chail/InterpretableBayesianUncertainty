#!/usr/bin/env python3
# coding: utf-8

from toolbox.load_dataset import load_mnist

from keras import backend as K
from keras.callbacks import Callback
from keras.models import Model
from keras import optimizers
from keras.utils import np_utils
import numpy as np
import os, pickle, sys
from timeit import default_timer as timer
from collections import defaultdict
from BBalpha_dropout import *

if len(sys.argv) != 3:
    print("Call this program like this:\n"
          "    ./mnist-mlp-train.py alpha run\n"
          "    e.g. ./mnist-mlp-train.py 0.5 1"
         )
    exit()

# extract command line arguments
alpha = float(sys.argv[1])
run = sys.argv[2]

# get dataset
train, validation, _ = load_mnist(flatten=True)

# constants
nb_train = train[0].shape[0]
nb_val = validation[0].shape[0]
input_dim = train[0].shape[1]
nb_classes = train[1].shape[1]

batch_size = 128
nb_layers = 2
nb_units = 100
p = 0.5
wd = 1e-6
dropout = 'MC'

K_mc = 10

epochs = 30

# model layers
input_shape = (input_dim, ) # (dimX, )
inp = Input(shape=input_shape)
layers = get_logit_mlp_layers(nb_layers, nb_units, p, wd, nb_classes, layers = [],
                              dropout=dropout)

# build model with MC samples
mc_logits = GenerateMCSamples(inp, layers, K_mc) # repeats stochastic layers K_mc times
# if alpha = 0.0, bbalpha returns categorical cross entropy with logits
loss_function = bbalpha_softmax_cross_entropy_with_mc_logits(alpha)
model = Model(inputs=inp, outputs=mc_logits)
# TODO: change the optimizer
opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=opt, loss=loss_function,
              metrics=['accuracy', loss_function, metric_avg_acc, metric_avg_ll])


train_Y_dup = np.squeeze(np.concatenate(K_mc * [train[1][:, None]], axis=1)) # N x K_mc x D
val_Y_dup = np.squeeze(np.concatenate(K_mc * [validation[1][:, None]], axis=1)) # N x K_mc x D

# training loop
directory = os.path.join('saved_models',
                         'mnist-mlp-alpha{}-run{}'.format(alpha, run))
os.makedirs(directory, exist_ok=True)
results = defaultdict(list)
max_acc = 0.
max_acc_ep = 0
ep = 0

while ep < max(2 * max_acc_ep, epochs):
# while ep < epochs:
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

    val_avg_acc = results['val_avg_acc'][-1]
    if val_avg_acc > max_acc:
        max_acc = val_avg_acc
        max_acc_ep = ep
        print("Updating max_acc_ep: {}".format(max_acc_ep))

        # save the model
        tic = timer()
        model.save(os.path.join(directory, 'model.h5'))
        toc = timer()
    print("Max_acc_ep: {}\t Max_acc: {:.3f}".format(max_acc_ep, max_acc))
    ep += 1

    # save result after every epoch
    with open(os.path.join(directory, 'results-2.p'), 'wb') as f:
        pickle.dump(results, f)
