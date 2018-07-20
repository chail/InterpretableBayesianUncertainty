import pickle
import numpy as np
from salience import utils_visualise as utlsV
import glob
import matplotlib.pyplot as plt
import string
from toolbox import load_dataset


(_, _, test) = load_dataset.load_image_data('cifar10', channels_first=True)
labels = load_dataset.load_image_labels('cifar10')
winSize=8

def increase_contrast(p):
    """ p is the salience map as an np.array"""
    return p**2 * np.sign(p) # better contrast visibility

def plot_original(ii, ax, x_test_im, y_true_class, y_pred_class, softmax):
    """ plot the original image and write true and predicted classes
        in the title on specified axis object
    """
    ax.set_title('({}) {}->{}\n{}%'
                          .format(string.ascii_uppercase[ii], 
                                  labels[y_true_class], labels[y_pred_class],
                                  int(round(softmax * 100))))
    ax.imshow(np.squeeze(x_test_im), interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    
def plot_salience(f, ax, p):    
    """ plot the salience map with colorbar on specified axes and 
        figure objects
        Input:
                f: figure object
                ax: axis object
                p: NxN salience map
    """
    im = ax.imshow(p, interpolation='nearest', cmap='bwr',
                            vmin=-np.max(np.abs(p)), vmax=np.max(np.abs(p)))
    ax.set_xticks([])
    ax.set_yticks([])
    
    ticks = [-np.max(np.abs(p)), 0, np.max(np.abs(p))]
    cbar = f.colorbar(im,ax=ax,fraction=0.046, pad=0.04, ticks=ticks)
    cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in ticks])  # vertically oriented colorbar
    

def plot_overlay(ax, x_test_im, p, contrast=False, cbar_lim=None):
    """ plot the salience map overlayed on image on specified axes
        Input:
                ax: axis object
                x_test_im: image without postprocessing
                p: NxN salience map
                contrast: whether to increase contrast of salience map
                    for better visualisation
    """
    if contrast:
        p = increase_contrast(p) # better contrast visibility
    p = utlsV.get_overlayed_image(x_test_im, p, 0.5, 0.8,
                                  cbar_lim=cbar_lim)
    if not cbar_lim:
        cbar_lim=np.max(np.abs(p))
    ax.imshow(p, interpolation='nearest', cmap='bwr',
              vmin=-cbar_lim, vmax=cbar_lim)
    ax.set_xticks([])
    ax.set_yticks([])
    
    
def set_left_axis_labels(axes, titles):
    """ sets the y-axis label on axes to the strings specified
        in titles list
    """
    assert len(axes) == len(titles)
    for ii in range(len(axes)):
        axes[ii][0].set_ylabel(titles[ii])
        
        
def plot_epistemic_aleatoric(idx, savepath=None):
    """ plots a 3 x len(idx) plot
        idx is a list of indices 
        top row is original image
        middle row is epistemic salience map
        bottom row is aleatoric salience map
    """
    f, axes = plt.subplots(3, len(idx), figsize=(12, 7))
    for ii, ind in enumerate(idx):
        x_test = test[0][ind]
        x_test_im = test[0][ind]
        y_true = np.argmax(test[1][ind])
        filename = glob.glob('../../salience/results-cifar10-bbalpha-run1/cifar10_test{}*.p'
                             .format(ind, winSize))
        results = pickle.load(open(filename[0], 'rb'))

        x_test_im = np.transpose(x_test_im, (1, 2, 0))
        imsize = x_test_im.shape[:-1]
        
        # original image
        plot_original(ii, axes[0][ii], x_test_im, y_true, results['y_pred'],
                      np.max(results['pred_outputs']['pred']))

        # epistemic
        plot_overlay(axes[1][ii], x_test_im, 
                     results['epistemic'].reshape(imsize[0], imsize[1]), 
                     contrast=True)

        # aleatoric
        plot_overlay(axes[2][ii], x_test_im, 
                     results['aleatoric']
                     .reshape(imsize[0], imsize[1]), 
                     contrast=True)
    set_left_axis_labels(axes, ['Original', 'Epistemic', 'Aleatoric'])
    if savepath:
        f.savefig(savepath, dpi=600)
        
        
def plot_epistemic_aleatoric_appendix(idx, savepath=None):
    f, axes = plt.subplots(3, len(idx), figsize=(12, 7))
    for ii, ind in enumerate(idx):
        x_test = test[0][ind]
        x_test_im = test[0][ind]
        y_true = np.argmax(test[1][ind])
        filename = glob.glob('../../salience/results-cifar10-bbalpha-run1/cifar10_test{}*.p'
                             .format(ind, winSize))
        results = pickle.load(open(filename[0], 'rb'))

        x_test_im = np.transpose(x_test_im, (1, 2, 0))
        imsize = x_test_im.shape[:-1]

        # original image
        plot_original(ii, axes[0][ii], x_test_im, y_true, results['y_pred'],
                      np.max(results['pred_outputs']['pred']))
        # epistemic
        plot_salience(f, axes[1][ii], results['epistemic'].reshape(imsize[0], imsize[1]))

        # aleatoric
        plot_salience(f, axes[2][ii], results['aleatoric'].reshape(imsize[0], imsize[1]))
    set_left_axis_labels(axes, ['Original', 'Epistemic', 'Aleatoric'])
    f.tight_layout()
    if savepath:
        f.savefig('appendix_' + savepath, dpi=600)
        
        
def plot_predictive_difference(idx, savepath=None):
    """ plots a 3 x len(idx) plot
        idx is a list of indices 
        top row is original image
        middle row is predictive uncertainty salience map
        bottom row is predictive difference salience map
    """
    f, axes = plt.subplots(3, len(idx), figsize=(12, 7))
    for ii, ind in enumerate(idx):
        x_test = test[0][ind]
        x_test_im = test[0][ind]
        y_true = np.argmax(test[1][ind])
        filename = glob.glob('../../salience/results-cifar10-bbalpha-run1/cifar10_test{}*.p'
                             .format(ind, winSize))
        results = pickle.load(open(filename[0], 'rb'))

        x_test_im = np.transpose(x_test_im, (1, 2, 0))
        imsize = x_test_im.shape[:-1]
        
        # original image
        plot_original(ii, axes[0][ii], x_test_im, y_true, results['y_pred'],
                      np.max(results['pred_outputs']['pred']))

        # predictive uncertainty
        plot_overlay(axes[1][ii], x_test_im, 
                     results['predictive'].reshape(imsize[0], imsize[1]), 
                     contrast=False)

        # predictive difference 
        plot_overlay(axes[2][ii], x_test_im, 
                     results['pred'][:, results['y_pred']]
                     .reshape(imsize[0], imsize[1]), 
                     contrast=False)
    set_left_axis_labels(axes, ['Original', 'Pred. Unc.', 'Pred. Diff.'])
    if savepath:
        f.savefig(savepath, dpi=600)
        
        
def plot_across_runs(ind, savepath=None):
    """ plots a 2 x 3 plot
        ind is an integer index
        top row is epistemic salience
        bottom row is aleatoric salience map
        each column is the salience map for a replicate of the model
        to show that maps across different replicates are similar
    """
    f, axes = plt.subplots(2,3, figsize=(9, 5))
    for ii in range(3): # across 3 runs
        x_test = test[0][ind]
        x_test_im = test[0][ind]
        y_true = np.argmax(test[1][ind])
        filename = glob.glob('../../salience/results-cifar10-bbalpha-run{}/cifar10_test{}*.p'
                             .format(ii+1,ind, winSize))
        results = pickle.load(open(filename[0], 'rb'))

        x_test_im = np.transpose(x_test_im, (1, 2, 0))
        imsize = x_test_im.shape[:-1]

        # epistemic
        plot_overlay(axes[0][ii], x_test_im, 
                     results['epistemic'].reshape(imsize[0], imsize[1]), 
                     contrast=True)
        # aleatoric
        plot_overlay(axes[1][ii], x_test_im, 
                     results['aleatoric'].reshape(imsize[0], imsize[1]), 
                     contrast=True)
    set_left_axis_labels(axes, ['Epistemic', 'Aleatoric'])
    axes[0][0].set_title('Run 1')
    axes[0][1].set_title('Run 2')
    axes[0][2].set_title('Run 3')

    if savepath:
        f.savefig('appendix_' + savepath, dpi=600)
    
    
def plot_training_n(ind, savepath=None):
    """ plots a 2 x 3 plot
        ind is an integer index
        top row is epistemic salience
        bottom row is aleatoric salience map
        each column is the salience map for a model trained on an increasing
        fraction of training data
    """

    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    fracs = ['frac{:.1f}'.format(i/10) for i in range(1, 11)]
    
    # determine colorbar limits (same limits used across each row)
    cbar_lim_epistemic = []
    cbar_lim_aleatoric = []
    for ii, f in enumerate(fracs):
        filename = [glob.glob('../../salience/results-cifar10-bbalpha-run{}/{}_cifar10_test{}*.p'
                              .format(r, f, ind, winSize)) for r in range(1, 4)]
        results = [pickle.load(open(f[0], 'rb')) for f in filename]
        diff = np.mean(np.concatenate([r['epistemic'] for r in results], axis=1), axis=1)
        cbar_lim_epistemic.append(np.max(np.abs(diff)))
        diff = np.mean(np.concatenate([r['aleatoric'] for r in results], axis=1), axis=1)
        cbar_lim_aleatoric.append(np.max(np.abs(diff)))
    
    cbar_lim_epistemic = increase_contrast(np.array(cbar_lim_epistemic))
    cbar_lim_aleatoric = increase_contrast(np.array(cbar_lim_aleatoric))
    cbar_lim_epistemic = np.max(cbar_lim_epistemic)
    cbar_lim_aleatoric = np.max(cbar_lim_aleatoric)
    
    for ii, f in enumerate(fracs):
        x_test = test[0][ind]
        x_test_im = test[0][ind]
        y_true = np.argmax(test[1][ind])

        filename = [glob.glob('../../salience/results-cifar10-bbalpha-run{}/{}_cifar10_test{}*.p'
                              .format(r, f, ind, winSize)) for r in range(1, 4)]
        results = [pickle.load(open(f[0], 'rb')) for f in filename]
        x_test_im = np.transpose(x_test_im, (1, 2, 0))
        imsize = x_test_im.shape[:-1]

        # epistemic
        diff = np.mean(np.concatenate([r['epistemic'] for r in results], axis=1), axis=1)
        p = diff.reshape(imsize[0], imsize[1])
        plot_overlay(axes[0][ii], x_test_im, p, contrast=True,
                     cbar_lim=cbar_lim_epistemic)
        axes[0][ii].set_title('{}%'.format((ii+1) * 10))

        # aleatoric
        diff = np.mean(np.concatenate([r['aleatoric'] for r in results], axis=1), axis=1)
        p = diff.reshape(imsize[0], imsize[1])
        plot_overlay(axes[1][ii], x_test_im, p, contrast=True,
                     cbar_lim=cbar_lim_aleatoric)

    set_left_axis_labels(axes, ['Epistemic', 'Aleatoric'])
    if savepath:
        fig.savefig(savepath, dpi=600)
