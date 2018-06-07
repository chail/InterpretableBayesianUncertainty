#!/usr/bin/env python3
# coding: utf-8

from toolbox.load_dataset import load_mnist

import tensorflow as tf
from timeit import default_timer as timer
from collections import defaultdict
from core_functions import *
import sys
import pickle

if len(sys.argv) != 3:
    print("Call this program like this:\n"
          "    ./mnist-cnn-train.py alpha run\n"
          "    e.g. ./mnist-cnn-train.py 0.5 1"
         )
    exit()

# extract command line arguments
alpha = float(sys.argv[1])
run = sys.argv[2]

# get dataset
train, validation, _ = load_mnist(flatten=False, channels_first=False)

# constants
assert train[0].shape[1] == train[0].shape[2], 'Input image not square'
input_size = train[0].shape[2]
in_channels = train[0].shape[3]
nb_train = train[0].shape[0]
nb_val = validation[0].shape[0]
input_dim = train[0].shape[1]
nb_classes = train[1].shape[1]

batch_size = 128
val_batch_size = nb_val
nb_layers = 2
nb_units = 100
p = 0.5
wd = 1e-6

K_mc = 10

learning_rate = 0.01
epochs = 30


# create the datasets with placeholder
batch_size_ph = tf.placeholder(tf.int64, name='batch_size_ph')
shuffle_ph = tf.placeholder(tf.int64, name='shuffle_ph')
features_data_ph = tf.placeholder(tf.float32,
                                  [None, input_size, input_size, in_channels],
                                  'features_data_ph')
labels_data_ph = tf.placeholder(tf.int32, [None, nb_classes], 'labels_data_ph')
dataset = tf.data.Dataset.from_tensor_slices((features_data_ph, labels_data_ph)).shuffle(shuffle_ph).batch(batch_size_ph)

# set up dataset iterator
iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                           dataset.output_shapes)
next_element = iterator.get_next()
dataset_init_op = iterator.make_initializer(dataset, name='dataset_init')


# build graph
x = next_element[0]
y = tf.cast(next_element[1], tf.float32)
y_dup = tf.stack([y for _ in range(K_mc)], axis=1, name='y_dup')
nb_elem = tf.identity(tf.shape(y)[0], name='nb_elem')


# trainable variables
nb_filters = 32
pool_size = 2
kernel_size = 3
flatten_dim = int(nb_filters * ((input_size - 4) / 2)**2)
variables = {}
with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
    weights = tf.get_variable('weight_variable', shape=(kernel_size,
                                                        kernel_size,
                                                        in_channels,
                                                        nb_filters),
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('bias_variable', shape=(nb_filters,),
                             initializer=tf.zeros_initializer())
    variables['wc1']=weights
    variables['bc1']=biases
with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
    weights = tf.get_variable('weight_variable', shape=(kernel_size,
                                                        kernel_size,
                                                        nb_filters,
                                                        nb_filters),
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('bias_variable', shape=(nb_filters,),
                             initializer=tf.zeros_initializer())
    variables['wc2']=weights
    variables['bc2']=biases
with tf.variable_scope('dense1', reuse=tf.AUTO_REUSE):
    weights = tf.get_variable('weight_variable', shape=(flatten_dim, nb_units),
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('bias_variable', shape=(nb_units,),
                             initializer=tf.zeros_initializer())
    variables['wd1']=weights
    variables['bd1']=biases
with tf.variable_scope('dense2', reuse=tf.AUTO_REUSE):
    weights = tf.get_variable('weight_variable', shape=(nb_units, nb_classes),
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('bias_variable', shape=(nb_classes,),
                             initializer=tf.zeros_initializer())
    variables['wd2']=weights
    variables['bd2']=biases


def build_mc_logit_network(nb_reps):
    """
    Build CNN with stochastic MC samples
    """
    conv1 = conv2d(x, variables['wc1'], variables['bc1'])
    conv1 = tf.nn.relu(conv1)
    conv2 = conv2d(conv1, variables['wc2'], variables['bc2'])
    conv2 = tf.nn.relu(conv2)
    maxpool = maxpool2d(conv2)
    flat = tf.reshape(maxpool, [-1, variables['wd1'].get_shape().as_list()[0]])
    layers = []
    layers.append([nn_layer(flat, 'dense1', p) for _ in range(nb_reps)])
    layers.append([nn_layer(h, 'dense2', p, act=tf.identity) for h in layers[-1]])
    mc_logits = tf.stack(layers[-1], axis=1, name='mc_logits')
    return mc_logits, layers


# graph output
mc_logits, layers = build_mc_logit_network(K_mc)
softmax_output = tf.nn.softmax(mc_logits, name='softmax_output')


# loss functions
if alpha != 0.0:
    loss = tf.identity(bbalpha_softmax_cross_entropy_with_mc_logits(mc_logits, y_dup, alpha), name='loss')
else:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_dup, logits=mc_logits), name='loss')


# regularization and optimizer step
regularizer = wd * tf.reduce_sum([tf.nn.l2_loss(variables[w]) for w in variables if 'w' in w ])
total_loss = tf.identity(loss + regularizer, name='total_loss')
train_step = tf.train.MomentumOptimizer(learning_rate, momentum=0.9,
                                        use_nesterov=True).minimize(total_loss)

# metrics on the train network (K_mc reps)
correct_prediction = tf.equal(tf.argmax(softmax_output, axis=-1), tf.argmax(y_dup, axis=-1), name='correct_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
softmax_avg = tf.reduce_mean(softmax_output, axis=1, name='softmax_avg')
avg_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(softmax_avg, axis=-1), 
                                               tf.argmax(y, axis=-1)), tf.float32), 
                              name='avg_accuracy')
avg_ll = tf.reduce_mean(tf.log(tf.reduce_sum(softmax_avg * y, axis=-1)), name='avg_ll')


# save model and accumulate results 
saver = tf.train.Saver()
results = defaultdict(list)


# training loop
max_acc = 0.
max_acc_ep = 0
ep = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # train until at least twice the time of max_acc_ep
    # while ep < max(2 * max_acc_ep, epochs):
    while ep < epochs:

        # initialize training run
        sess.run(dataset_init_op, feed_dict={
            features_data_ph: train[0],
            labels_data_ph: train[1],
            shuffle_ph: nb_train,
            batch_size_ph: batch_size
        })
        accum_n = 0.
        accum_total_loss = 0.
        accum_bb_loss = 0.
        accum_avg_acc = 0.
        accum_acc = 0.
        accum_ll = 0.
        tic = timer()

        # cycle through all training batches
        while True:
            try:
                metrics = sess.run([train_step, nb_elem, total_loss, loss,
                                    accuracy, avg_accuracy, avg_ll])
                n = metrics[1]
                accum_n += metrics[1]
                nb_batches_train = nb_train / n
                accum_total_loss += metrics[2] / nb_batches_train
                accum_bb_loss += metrics[3] / nb_batches_train
                accum_acc += metrics[4] / nb_batches_train
                accum_avg_acc += metrics[5] / nb_batches_train
                accum_ll += metrics[6] / nb_batches_train
            except tf.errors.OutOfRangeError:
                assert accum_n == nb_train, "Incomplete Training Epoch"
                break

        # accumulate training results
        toc = timer()
        results['train_N'].append(accum_n)
        results['train_total_loss'].append(accum_total_loss)
        results['train_bbalpha_loss'].append(accum_bb_loss)
        results['train_acc'].append(accum_acc)
        results['train_avg_acc'].append(accum_avg_acc)
        results['train_ll'].append(accum_ll)
        results['train_time'].append(toc-tic)
        print("Train Epoch: {}\tBB loss: {:.3f}\tAcc: {:.3f}\tAvg Acc: {:.3f}\tLL: {:.3f}\tTime: {:.3f}"
             .format(ep, accum_bb_loss, accum_acc, accum_avg_acc, accum_ll, toc-tic))

        # initialize validation run
        sess.run(dataset_init_op, feed_dict={
            features_data_ph: validation[0],
            labels_data_ph: validation[1],
            shuffle_ph: nb_val,
            batch_size_ph: val_batch_size
        })
        accum_n = 0.
        accum_total_loss = 0.
        accum_bb_loss = 0.
        accum_avg_acc = 0.
        accum_acc = 0.
        accum_ll = 0.
        tic = timer()

        # cycle through all validation batches
        while True:
            try:
                # note that validation run is still using the TRAIN network (K_mc reps)
                metrics = sess.run([nb_elem, total_loss, loss,
                                    accuracy, avg_accuracy, avg_ll])
                n = metrics[0]
                accum_n += metrics[0]
                nb_batches_val = nb_val / n
                accum_total_loss += metrics[1] / nb_batches_val
                accum_bb_loss += metrics[2] / nb_batches_val
                accum_acc += metrics[3] / nb_batches_val
                accum_avg_acc += metrics[4] / nb_batches_val
                accum_ll += metrics[5] / nb_batches_val
            except tf.errors.OutOfRangeError:
                assert accum_n == nb_val, "Incomplete Validation Run"
                break

        # accumulate validation results
        toc = timer()
        results['val_N'].append(accum_n)
        results['val_total_loss'].append(accum_total_loss)
        results['val_bbalpha_loss'].append(accum_bb_loss)
        results['val_acc'].append(accum_acc)
        results['val_avg_acc'].append(accum_avg_acc)
        results['val_ll'].append(accum_ll)
        results['val_time'].append(toc-tic)
        print("Val Epoch: {}\tBB loss: {:.3f}\tAcc: {:.3f}\tAvg Acc: {:.3f}\tLL: {:.3f}\tTime: {:.3f}"
             .format(ep, accum_bb_loss, accum_acc, accum_avg_acc, accum_ll, toc-tic))

        # update patience parameters
        if accum_avg_acc > max_acc:
            max_acc = accum_avg_acc
            max_acc_ep = ep
            print("Updating max_acc_ep: {}".format(max_acc_ep))

            # save the model
            tic = timer()
            save_meta = True if ep == 0 else False
            save_path = saver.save(sess,
                                   "./saved_models/mnist-cnn-alpha{}-run{}/model"
                                   .format(alpha, run), write_meta_graph=save_meta)
            toc = timer()
            print("Model saved in path: {}, Time: {:.3f}".format(save_path, toc-tic))

        print("Max_acc_ep: {}\t Max_acc: {:.3f}".format(max_acc_ep, max_acc))
        ep += 1

        # save result after every epoch
        with open('./saved_models/mnist-cnn-alpha{}-run{}/results.p'
                  .format(alpha, run), 'wb') as f:
            pickle.dump(results, f)
