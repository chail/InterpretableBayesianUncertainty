import pickle
import numpy as np
from salience import utils_visualise as utlsV
import glob
import matplotlib.pyplot as plt
import string
from toolbox import load_dataset
import os

class visualiser:
    
    def __init__(self, dataset):
        self.dataset = dataset
        if dataset == 'cifar10':
            (_, _, test) = load_dataset.load_image_data('cifar10', channels_first=True)
            self.labels = load_dataset.load_image_labels('cifar10')
            self.winSize=8
            self.test = test
            self.pathroot = os.path.join('..', '..', 'salience', 'results-cifar10-bbalpha-run1')
            self.file = 'cifar10_test{}*_winSize{}*.p'
            self.nruns = 3
            self.runs = [os.path.join('..', '..', 'salience', 'results-cifar10-bbalpha-run{}'
                                     .format(r)) for r in range(1, self.nruns+1)]
        elif dataset == 'isic':
            (_, _, test) = load_dataset.load_image_data('isic', channels_first=True)
            self.winSize=20
            self.labels = load_dataset.load_image_labels('isic')
            self.test = test
            self.pathroot = os.path.join('..', '..', 'salience', 'results-isic-bbalpha-run1')
            self.file = 'isic_test{}*_winSize{}*.p'
            self.nruns = 1
            self.runs = [os.path.join('..', '..', 'salience', 'results-isic-bbalpha-run{}'
                                     .format(r)) for r in range(1, self.nruns+1)]
            
                                      
    def increase_contrast(self, p):
        """ p is the salience map as an np.array"""
        return p**2 * np.sign(p) # better contrast visibility

    def plot_original(self, ii, ax, x_test_im, y_true_class, y_pred_class, softmax):
        """ plot the original image and write true and predicted classes
            in the title on specified axis object
        """
        ax.set_title('({}) {}->{}\n{}%'
                              .format(string.ascii_uppercase[ii], 
                                      self.labels[y_true_class], self.labels[y_pred_class],
                                      int(round(softmax * 100))))
        ax.imshow(np.squeeze(x_test_im), interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])


    def plot_salience(self, f, ax, p):    
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


    def plot_overlay(self, ax, x_test_im, p, contrast=False, cbar_lim=None):
        """ plot the salience map overlayed on image on specified axes
            Input:
                    ax: axis object
                    x_test_im: image without postprocessing
                    p: NxN salience map
                    contrast: whether to increase contrast of salience map
                        for better visualisation
        """
        if contrast:
            p = self.increase_contrast(p) # better contrast visibility
        p = utlsV.get_overlayed_image(x_test_im, p, 0.5, 0.8,
                                      cbar_lim=cbar_lim)
        if not cbar_lim:
            cbar_lim=np.max(np.abs(p))
        ax.imshow(p, interpolation='nearest', cmap='bwr',
                  vmin=-cbar_lim, vmax=cbar_lim)
        ax.set_xticks([])
        ax.set_yticks([])


    def set_left_axis_labels(self, axes, titles, size=None):
        """ sets the y-axis label on axes to the strings specified
            in titles list
        """
        assert len(axes) == len(titles)
        for ii in range(len(axes)):
            if size:
                axes[ii][0].set_ylabel(titles[ii], fontsize=size)
            else:
                axes[ii][0].set_ylabel(titles[ii])


    def plot_epistemic_aleatoric(self, idx, savepath=None, figsize=(12, 7), dataset=None):
        """ plots a 3 x len(idx) plot
            idx is a list of indices 
            top row is original image
            middle row is epistemic salience map
            bottom row is aleatoric salience map
        """
        f, axes = plt.subplots(3, len(idx), figsize=figsize)
        for ii, ind in enumerate(idx):
            x_test = self.test[0][ind]
            x_test_im = self.test[0][ind]
            y_true = np.argmax(self.test[1][ind])
            filename = glob.glob(os.path.join(self.pathroot, self.file.format(ind, self.winSize)))
            results = pickle.load(open(filename[0], 'rb'))
            x_test_im = np.transpose(x_test_im, (1, 2, 0))
            imsize = x_test_im.shape[:-1]

            # original image
            self.plot_original(ii, axes[0][ii], x_test_im, y_true, results['y_pred'],
                               np.max(results['pred_outputs']['pred']))

            # epistemic
            self.plot_overlay(axes[1][ii], x_test_im, 
                              results['epistemic'].reshape(imsize[0], imsize[1]), 
                              contrast=True)

            # aleatoric
            self.plot_overlay(axes[2][ii], x_test_im, 
                              results['aleatoric']
                              .reshape(imsize[0], imsize[1]), 
                              contrast=True)
        self.set_left_axis_labels(axes, ['Original', 'Epistemic', 'Aleatoric'])
        if savepath:
            f.savefig(savepath, dpi=600, bbox_inches='tight')


    def plot_epistemic_aleatoric_appendix(self, idx, savepath=None):
        f, axes = plt.subplots(3, len(idx), figsize=(12, 7))
        for ii, ind in enumerate(idx):
            x_test = self.test[0][ind]
            x_test_im = self.test[0][ind]
            y_true = np.argmax(self.test[1][ind])
            filename = glob.glob(os.path.join(self.pathroot, self.file.format(ind, self.winSize)))
            results = pickle.load(open(filename[0], 'rb'))

            x_test_im = np.transpose(x_test_im, (1, 2, 0))
            imsize = x_test_im.shape[:-1]

            # original image
            self.plot_original(ii, axes[0][ii], x_test_im, y_true, results['y_pred'],
                               np.max(results['pred_outputs']['pred']))
            # epistemic
            self.plot_salience(f, axes[1][ii], results['epistemic'].reshape(imsize[0], imsize[1]))

            # aleatoric
            self.plot_salience(f, axes[2][ii], results['aleatoric'].reshape(imsize[0], imsize[1]))
        self.set_left_axis_labels(axes, ['Original', 'Epistemic', 'Aleatoric'])
        f.tight_layout()
        if savepath:
            f.savefig('appendix_' + savepath, dpi=600, bbox_inches='tight')


    def plot_predictive_difference(self, idx, savepath=None):
        """ plots a 3 x len(idx) plot
            idx is a list of indices 
            top row is original image
            middle row is predictive uncertainty salience map
            bottom row is predictive difference salience map
        """
        f, axes = plt.subplots(3, len(idx), figsize=(12, 7))
        for ii, ind in enumerate(idx):
            x_test = self.test[0][ind]
            x_test_im = self.test[0][ind]
            y_true = np.argmax(self.test[1][ind])
            filename = glob.glob(os.path.join(self.pathroot, self.file.format(ind, self.winSize)))
            results = pickle.load(open(filename[0], 'rb'))

            x_test_im = np.transpose(x_test_im, (1, 2, 0))
            imsize = x_test_im.shape[:-1]

            # original image
            self.plot_original(ii, axes[0][ii], x_test_im, y_true, results['y_pred'],
                               np.max(results['pred_outputs']['pred']))

            # predictive uncertainty
            self.plot_overlay(axes[1][ii], x_test_im, 
                              results['predictive'].reshape(imsize[0], imsize[1]), 
                              contrast=False)

            # predictive difference 
            self.plot_overlay(axes[2][ii], x_test_im, 
                              results['pred'][:, results['y_pred']] # y_true] 
                              .reshape(imsize[0], imsize[1]), 
                              contrast=False)
        self.set_left_axis_labels(axes, ['Original', 'Pred. Unc.', 'Pred. Diff.'])
        if savepath:
            f.savefig(savepath, dpi=600, bbox_inches='tight')


    def plot_across_runs(self, ind, savepath=None):
        """ plots a 2 x 3 plot
            ind is an integer index
            top row is epistemic salience
            bottom row is aleatoric salience map
            each column is the salience map for a replicate of the model
            to show that maps across different replicates are similar
        """
        f, axes = plt.subplots(2,3, figsize=(9, 5))
        for ii in range(self.nruns): # across 3 runs
            x_test = self.test[0][ind]
            x_test_im = self.test[0][ind]
            y_true = np.argmax(self.test[1][ind])
            filename = glob.glob(os.path.join(self.runs[ii], self.file.format(ind, self.winSize)))
            results = pickle.load(open(filename[0], 'rb'))

            x_test_im = np.transpose(x_test_im, (1, 2, 0))
            imsize = x_test_im.shape[:-1]

            # epistemic
            self.plot_overlay(axes[0][ii], x_test_im, 
                              results['epistemic'].reshape(imsize[0], imsize[1]), 
                              contrast=True)
            # aleatoric
            self.plot_overlay(axes[1][ii], x_test_im, 
                              results['aleatoric'].reshape(imsize[0], imsize[1]), 
                              contrast=True)
        self.set_left_axis_labels(axes, ['Epistemic', 'Aleatoric'])
        axes[0][0].set_title('Run 1')
        axes[0][1].set_title('Run 2')
        axes[0][2].set_title('Run 3')

        if savepath:
            f.savefig('appendix_' + savepath, dpi=600, bbox_inches='tight')


    def plot_training_n(self, ind, savepath=None, target=None):
        """ plot the epistemic and aleatoric salience of ind as more
        training data is provided
        target is the unbalanced class (use None for all balanced classes)
        """
        
        # only works for cifar10
        assert self.dataset == 'cifar10'
        
        fig, axes = plt.subplots(2, 10, figsize=(15, 3))
        fracs = ['frac{:.1f}'.format(i/10) for i in range(1, 11)]

        # determine colorbar limits (same limits used across each row)
        cbar_lim_epistemic = []
        cbar_lim_aleatoric = []
        for ii, f in enumerate(fracs):
            if target is not None:
                filename = [glob.glob(os.path.join(r, 'target{}_'.format(target)
                                                   + f + '_' + self.file.format(ind, self.winSize)))
                            for r in self.runs]
            else:
                filename = [glob.glob(os.path.join(r, f + '_' + self.file.format(ind, self.winSize)))
                            for r in self.runs]
            results = [pickle.load(open(f[0], 'rb')) for f in filename]
            diff = np.mean(np.concatenate([r['epistemic'] for r in results], axis=1), axis=1)
            cbar_lim_epistemic.append(np.max(np.abs(diff)))
            diff = np.mean(np.concatenate([r['aleatoric'] for r in results], axis=1), axis=1)
            cbar_lim_aleatoric.append(np.max(np.abs(diff)))

        cbar_lim_epistemic = self.increase_contrast(np.array(cbar_lim_epistemic))
        cbar_lim_aleatoric = self.increase_contrast(np.array(cbar_lim_aleatoric))
        cbar_lim_epistemic = np.max(cbar_lim_epistemic)
        cbar_lim_aleatoric = np.max(cbar_lim_aleatoric)

        for ii, f in enumerate(fracs):
            x_test = self.test[0][ind]
            x_test_im = self.test[0][ind]
            y_true = np.argmax(self.test[1][ind])
            if target is not None:
                filename = [glob.glob(os.path.join(r, 'target{}_'.format(target)
                                                   + f + '_' + self.file.format(ind, self.winSize)))
                            for r in self.runs]
            else:
                filename = [glob.glob(os.path.join(r, f + '_' + self.file.format(ind, self.winSize)))
                            for r in self.runs]
            results = [pickle.load(open(f[0], 'rb')) for f in filename]
            x_test_im = np.transpose(x_test_im, (1, 2, 0))
            imsize = x_test_im.shape[:-1]

            # epistemic
            diff = np.mean(np.concatenate([r['epistemic'] for r in results], axis=1), axis=1)
            p = diff.reshape(imsize[0], imsize[1])
            self.plot_overlay(axes[0][ii], x_test_im, p, contrast=True,
                              cbar_lim=cbar_lim_epistemic)
            percentage=int(round(np.max(results[0]['pred_outputs']['pred']) * 100))
            # uncomment to show softmax percentage and predicted class
            #axes[0][ii].set_title('{}%\n{}\n{}%'.format((ii+1) * 10, self.labels[results[0]['y_pred']],
            #                                            percentage), fontsize=15)
            axes[0][ii].set_title('{}%'.format((ii+1)*10), fontsize=15)

            # aleatoric
            diff = np.mean(np.concatenate([r['aleatoric'] for r in results], axis=1), axis=1)
            p = diff.reshape(imsize[0], imsize[1])
            self.plot_overlay(axes[1][ii], x_test_im, p, contrast=True,
                              cbar_lim=cbar_lim_aleatoric)
            

        self.set_left_axis_labels(axes, ['Epistemic', 'Aleatoric'], 15)
        if savepath:
            fig.savefig(savepath, dpi=600, bbox_inches='tight')
            
    def plot_training_n_uncertainty(self, ind, savepath=None, target=None):
        """ plot the epistemic and aleatoric uncertainty of ind as more
        training data is provided
        target is the unbalanced class (use None for all balanced classes)
        """
        
        # only works for cifar10
        assert self.dataset == 'cifar10'
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 3))
        fracs = ['frac{:.1f}'.format(i/10) for i in range(1, 11)]
        
        x_test = self.test[0][ind]
        x_test_im = self.test[0][ind]
        x_test_im = np.transpose(x_test_im, (1, 2, 0))
        y_true = np.argmax(self.test[1][ind])
        axes[0].imshow(np.squeeze(x_test_im), interpolation='nearest')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title('Test Image', fontsize=15)
        
        epistemic = []
        aleatoric = []
        for ii, f in enumerate(fracs):
            if target is not None:
                filename = [glob.glob(os.path.join(r, 'target{}_'.format(target)
                                                   + f + '_' + self.file.format(ind, self.winSize)))
                            for r in self.runs]
            else:
                filename = [glob.glob(os.path.join(r, f + '_' + self.file.format(ind, self.winSize)))
                            for r in self.runs]
            results = [pickle.load(open(f[0], 'rb')) for f in filename]

            epistemic.append([r['pred_outputs']['epistemic'][0] for r in results])
            aleatoric.append([r['pred_outputs']['aleatoric'][0] for r in results])
            
        epistemic = np.array(epistemic)
        aleatoric = np.array(aleatoric)
        
        print(epistemic.shape)
        xaxis = [i/10 for i in range(1, 11)]
        axes[1].errorbar(xaxis, np.mean(epistemic, axis=1), np.std(epistemic, axis=1))
        axes[2].errorbar(xaxis, np.mean(aleatoric, axis=1), np.std(aleatoric, axis=1))
        axes[1].set_xlabel('Fraction of target class', fontsize=15)
        axes[2].set_xlabel('Fraction of target class', fontsize=15)
        axes[1].set_ylabel('Epistemic Unc.', fontsize=15)
        axes[2].set_ylabel('Aleatoric Unc.', fontsize=15)

        if savepath:
            fig.savefig(savepath, dpi=600, bbox_inches='tight')
