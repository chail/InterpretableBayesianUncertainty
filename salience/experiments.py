# -*- coding: utf-8 -*-

# the following is needed to avoid some error that can be thrown when 
# using matplotlib.pyplot in a linux shell
import matplotlib
matplotlib.use('Agg')

# standard imports
import numpy as np
import time
import os
import pickle
import glob

# most important script - relevance estimator
from compute_salience import UncertaintySalienceAnalyser

# utilities
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import utils_classifiers as utlC
import utils_sampling as utlS
import utils_visualise as utlV
from toolbox import load_dataset

# otherwise TF grabs all available gpu memory
if not hasattr(K, "tf"):
    raise RuntimeError("This code requires keras to be configured"
                       " to use the TensorFlow backend.")

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))
# ------------------------ CONFIGURATION ------------------------
# -------------------> CHANGE SETTINGS HERE <--------------------

# pick a dataset to use: ['mnist', 'cifar10', 'svhn', 'isic']
dataset = 'cifar10'

# path to saved tf graph with uncertainties
netname = 'bbalpha-run1'  # directory used for saving

# optional suffix string is appended to each saved file
# use None for no suffix
suffix = 'random'

# specify the model to use
path_to_model = '../models/bbalpha/keras/saved_models/' +\
        '{}-cnn-alpha0.5-run1/model-test.h5'.format(dataset)

# pick test image indices which are analysed
# (if None, all images will be analysed)
test_indices = list(np.random.choice(10000, 1))

# window size (i.e., the size of the pixel patch that is marginalised out in each step)
win_size = 8                 # k in alg 1 (see paper)

# indicate whether windows should be overlapping or not
overlapping = True

# settings for sampling 
sampl_style = 'conditional' # (only conditional is implemented here)
num_samples = 10
padding_size = 2            # important for conditional sampling,
                            # l = win_size+2*padding_size in alg 1
                            # (see paper)

# mode is 'BNN' or 'DNN'
# 'DNN' mode will only compute predictive difference
# 'BNN' mode also computes visualisations for 3 types of uncertainty
mode = 'DNN'

# for different networks, implement a new constructor in utlC
# to load a trained  model and perform a forward pass
# note: for DNN mode, the output of a forward pass should have dimensions NxC
# and for BNN mode shold be NxKxC where K is number of weight samples and C is
# number of classes
net = utlC.BBalpha_keras_net(path_to_model)

# salience visualisations require images of dimension (n, channels, w, h)
# you can modify this function to reshape the image into the dimensions
# required by the network via np.transpose or np.reshape
def target_func(x):
    # x has dimensions (n, channels, w, h)
    # net expects input dimensions (n, w, h, channels)
    # need to first transpose the dimensions of x
    x = np.transpose(x, (0, 2, 3, 1))
    assert np.ndim(x) == 4, 'Expected x to have 4 dim shape (n, w, h, channels)'
    assert x.shape[-1] == 1 or x.shape[-1] == 3, 'Expected 1 or 3 channels'
    return net.forward_pass(x)


# ------------------------ SET-UP ------------------------

# load data
(train, val, test) = load_dataset.load_image_data(dataset, channels_first=True)
labels = load_dataset.load_image_labels(dataset)

if not test_indices:
    test_indices = [i for i in range(test[0].shape[0])]

# make folder for saving the results if it doesn't exist
results_dir = './results-{}-{}/'.format(dataset, netname)
os.makedirs(results_dir, exist_ok=True)

# check mode
assert mode == 'DNN' or mode == 'BNN', \
        "invalid mode, use 'DNN' or 'BNN' only"

# ------------------------ EXPERIMENTS ------------------------

# for the given test indices, do the prediction difference analysis
for test_idx in test_indices:

    # get the specific image (preprocessed, can be used as input to the target function)
    x_test = test[0][test_idx]
    # get the image for plotting (not preprocessed)
    x_test_im = test[0][test_idx]
    # prediction of the network
    y_mc = target_func(np.expand_dims(x_test, axis=0))
    y_avg = np.mean(y_mc, axis=1)
    y_pred = np.argmax(y_avg)

    y_true = np.argmax(test[1][test_idx])
    print("Test Image: {}\tPredicted: {}\tTrue:{}"
          .format(test_idx, y_pred, np.argmax(test[1][test_idx])))

    # get the path for saving the results
    if sampl_style == 'conditional':
        filename = '{}_test{}_winSize{}_condSampl_numSampl{}_padSize{}'\
                .format(dataset, test_idx, win_size, num_samples, padding_size)
        path_base = os.path.join(results_dir, filename)
#    elif sampl_style == 'marginal':
#        filename = '{}_test{}_winSize{}_margSampl_numSampl{}'\
#                .format(dataset, test_idx, win_size, num_samples)
#        path_base = os.path.join(results_dir, filename)

    # add DNN mode to path_base
    if mode == 'DNN':
        path_base = path_base + '_DNN'

    # add optional suffix to path_base
    save_path = path_base + '_' + suffix if suffix else path_base
    print(save_path)

    existing_results = glob.glob(path_base + '*.p')
    if os.path.exists(save_path+'.p'):
        print('Results for test{} exist, will move to the next'
              ' image.'.format(test_idx))
    elif existing_results:
        print('Results for test{} exist under a different suffix'
              .format(test_idx))
        print('Linking {} to {}'.format(existing_results[0], save_path+'.p'))
        os.system('ln -s {} {}'.format(existing_results[0], save_path+'.p'))
    else:
        print("Analysing...test image: {}, net: {}, window size: {}, sampling: {}"
              .format(test_idx, netname, win_size, sampl_style))

        start_time = time.time()

        if sampl_style == 'conditional':
            sampler = utlS.cond_sampler(win_size=win_size,
                                        padding_size=padding_size,
                                        X=train[0], directory=results_dir)
#        elif sampl_style == 'marginal':
#            sampler = utlS.marg_sampler(win_size=win_size, X=train[0])

        analyser = UncertaintySalienceAnalyser(x_test, target_func, sampler,
                                               num_samples=num_samples,
                                               mode=mode)
        salience_dict, counts = analyser.get_rel_vect(win_size=win_size,
                                                      overlap=overlapping)

        salience_dict['y_pred'] = y_pred
        salience_dict['pred_outputs'] = analyser.true_tar_val

        with open(save_path + '.p', 'wb') as f:
            pickle.dump(salience_dict, f)

        print("--- Total computation took {:.4f} seconds  ---".format((time.time() -
                                                                      start_time)))
    # plot and save the results
    existing_results = glob.glob(path_base + '*.png')
    if os.path.exists(save_path+'.png'):
        print('Relevance map for test{} exist, will move to the next'
              ' image.'.format(test_idx))
    elif existing_results:
        print('Results for test{} exist under a different suffix'
              .format(test_idx))
        print('Linking {} to {}'.format(existing_results[0], save_path+'.png'))
        os.system('ln -s {} {}'.format(existing_results[0], save_path+'.png'))
    else:
        print("Plotting...test image: {}, net: {}, window size: {}, sampling: {}"
              .format(test_idx, netname, win_size, sampl_style))
        salience_dict = pickle.load(open(save_path + '.p', 'rb'))
        if mode == 'BNN':
            titles = ['epistemic', 'aleatoric', 'predictive']
            diffs = [salience_dict[x] for x in titles]
            titles.append('pred')
            diffs.append(salience_dict['pred'][:, salience_dict['y_pred']])
        elif mode == 'DNN':
            titles = labels
            diffs = [salience_dict['pred'][:, ii] for ii in range(len(labels))]
        utlV.plot_results(x_test_im, y_true, salience_dict['y_pred'],
                          diffs, titles, labels,
                          save_path + '.png')

