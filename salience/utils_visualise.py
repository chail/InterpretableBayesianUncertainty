# -*- coding: utf-8 -*-
"""
Some utility functions for visualisation
"""

from skimage import color
import numpy as np
import matplotlib.pyplot as plt


def plot_results(x_test_im, y_true, y_pred, diff_list, diff_labels,
                 classnames=None, save_path=None, true_classnames=None,
                 cmap='seismic'):
    '''
    Plot the results of the relevance estimation
    Input:
            x_test_im       image in plotting format (without normalisation)
            y_true          true class label (an integer)
            y_pred          predicted class label (an integer)
            diff_list       list of predictive differences, each element of the
                            list should be an (im_height*im_width,) shape numpy
                            array
            diff_labels     list of strings indicating the type of difference,
                            e.g. probability, epistemic uncertainty, etc.
            classnames      list of strings indicating correspondence between
                            class index integer and the class name
            save_path       saves the image, if provided
            true_classnames list of strings indicating correspondence between
                            class index integer and the class name for the true
                            class (useful if true classes differs from the
                            predicted classes, like if network trained to
                            predict cifar10 is provided cifar100 images as
                            input)
            cmap            string for colormap, 'seismic' or 'bwr' works well
    '''

    # check input, make sure it is channels first
    assert np.ndim(x_test_im) == 3, 'expected 3 dim input'
    assert x_test_im.shape[0] == 1 or x_test_im.shape[0] == 3,\
            'expected channels first'

    n_outputs = len(diff_list)
    assert len(diff_labels) == n_outputs,\
            'expected diff_list and diff_labels to be same length'

    # switch to channels last for visualisation
    x_test_im = np.transpose(x_test_im, (1, 2, 0))
    imsize = x_test_im.shape[:-1]

    # print classnames
    print('True: {}, Pred: {}'.format(y_true, y_pred))
    if true_classnames and classnames:
        print('True: {}, Pred: {}'.format(true_classnames[y_true],
                                          classnames[y_pred]))
    elif classnames:
        print('True: {}, Pred: {}'.format(classnames[y_true],
                                          classnames[y_pred]))



    f, axes = plt.subplots(n_outputs, 3, figsize=(10, 3*n_outputs))
    # want axes to be an n_outputs x 3 dimension array
    if np.ndim(axes) == 1:
        axes = np.expand_dims(axes, axis=0)

    # squeeze to remove singleton dim in mnist
    for (ax, diff, lab) in zip(axes, diff_list, diff_labels):
        ax[0].imshow(np.squeeze(x_test_im), interpolation='nearest')
        ax[0].set_title('Original')
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        assert diff.shape[0] == imsize[0] * imsize[1],\
                'unexpected size of relevance vector'

        p = diff.reshape(imsize[0], imsize[1])
        im = ax[1].imshow(p, interpolation='nearest', cmap=cmap,
                     vmin=-np.max(np.abs(p)), vmax=np.max(np.abs(p)))
        ax[1].set_title('Diff ' + lab)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        f.colorbar(im,ax=ax[1],fraction=0.046, pad=0.04)
        p = get_overlayed_image(x_test_im, p)
        ax[2].imshow(p, interpolation='nearest', cmap=cmap,
                     vmin=-np.max(np.abs(p)), vmax=np.max(np.abs(p)))
        ax[2].set_title('overlay')
        ax[2].set_xticks([])
        ax[2].set_yticks([])

    if save_path:
        f.savefig(save_path)
        plt.close(f)

def get_overlayed_image(x, c, gray_factor_bg=0.3, alpha=0.6, cmap='seismic',
                        cbar_lim=None):
    '''
    For an image x and a relevance vector c, overlay the image with the
    relevance vector to visualise the influence of the image pixels.
    '''
    imDim = x.shape[0]

    if np.ndim(c)==1:
        c = c.reshape((imDim,imDim))
    if x.shape[-1] == 1: # this happens with the MNIST Data
        x = 1-np.dstack((x, x, x))*gray_factor_bg # invert and make it grayish
    if x.shape[-1] == 3: # colour images
        x = color.rgb2gray(x)
        x = 1-(1-x)*gray_factor_bg
        x = np.dstack((x,x,x))

    # Construct a colour image to superimpose
    if not cbar_lim:
        cbar_lim = np.max(np.abs(c))
    im = plt.imshow(c, cmap=cmap, vmin=-cbar_lim, vmax=cbar_lim,
                    interpolation='nearest')
    color_mask = im.to_rgba(c)[:,:,[0,1,2]] # omit alpha channel

    # Convert the input image and color mask to Hue Saturation Value (HSV) colorspace
    img_hsv = color.rgb2hsv(x)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)

    return img_masked

