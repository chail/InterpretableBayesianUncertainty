# -*- coding: utf-8 -*-

import numpy as np
from numpy import ma
import time

# TODO: cleanup
import matplotlib.pyplot as plt
import pickle

# TODo: change batch size,  _evaluate_prediction_difference(self, tarVals),
# tarVals, test_per_batch, num_blobs, return a dictionary?

# TODO: clean up comments

class UncertaintySalienceAnalyser:
    '''
    This class computes how uncertainty changes if we know a feature of an
    image versus when we do not know that feature. It is based off of
    the prediction difference analysis approach, i.e., a method
    which estimates how important the individual input features are to a
    (already trained) classifier, given a specific input to the classifier.
    To this end, a relevance is estimated which is of the same size as the
    input and reflects the importance of each feature.

    This version is updated to work with grayscale images and RGB colour
    images. It is assumed that the colour channels are along axis 0, such that
    the input image has dimensions (channels, width, height)

    This pipeline was heavily modified from the approach in:
        https://github.com/lmzintgraf/DeepVis-PredDiff
    Credit goes to the original authors.

    '''

    def __init__(self, x, tar_func, sampler, num_samples=10):
        '''
        Input:
            x           the feature vector for which we want to make the analysis (can be a hidden layer!)
                        Has to be numpy array of the dimension that fits to targetFunc
            tar_func    the target function, can be the output of classifier or intermediate layer
                        (must take x as input, keep this in mind when starting at intermediate layers!)
            num_samples the number of samples used for marginalising out features
            batch_size  batch size for caffe network (in tar_func)
            prob_tar    boolean, indicates if the target values are probabilities
                        (not necessarily the case when we look at hidden nodes)
        '''

        # inputs
        assert np.ndim(x) == 3, 'expected 3 dimensional x (channels, w, h)'
        assert x.shape[0] == 3 or x.shape[0] == 1, 'expected 3 or 1 channels'
        self.x = np.copy(x)
        self.tar_func = tar_func
        self.sampler = sampler
        self.num_samples = num_samples
        self.channels = x.shape[0]
        self.n_images = 0

        # analysis is done for all channels at once, so divide number of total
        # features by the number of channels
        self.num_feats = int(len(self.x.ravel())/self.channels)

        # this is used to determine the dimensions of the output in
        # get_rel_vect, however, it is not actually used in saliency analysis
        # (the same weights are used for all inputs in the saliency analysis)
        x_in = np.expand_dims(self.x, axis=0)
        pred_mc = self.tar_func(x_in)
        self.temp_tar_val = self.compute_uncertainties(pred_mc)
        # drop the first dimension of the elements in the true target value list,
        # since it is not necessary (since we only forwarded a single feature vector)
        self.temp_tar_val = {key:self.temp_tar_val[key][0]
                             if np.ndim(self.temp_tar_val[key]) > 1
                             else self.temp_tar_val[key]
                             for key in self.temp_tar_val}
#%%
# -------------------- METHOD COMPUTING UNCERTAINTIES--------------------------  

    def compute_uncertainties(self, pred_mc):
        """
        pred_mc is an N x K x C matrix
            N is the number of samples
            K is the number of draws from the posterior weight distribution
            C is the number of classes in the prediction
        Returns: a dictionary containing
            pred:   predictive categorical softmax obtained by integrating over
            draws from the weights, of shape (N, C)
            aleatoric:  aleatoric uncertainty, of shape (N,)
            epistemic:  epistemic uncertainty, of shape (N,)
            predictive:  predictive uncertainty, of shape (N,)
        """

        nb_test = pred_mc.shape[1]
        pred = np.mean(pred_mc,axis=1)
        predictive_uncertainty = - np.sum(pred * ma.log2(pred).filled(0), axis=-1)
        aleatoric_uncertainty = - 1/nb_test \
                * np.sum(pred_mc * ma.log2(pred_mc).filled(0), axis=(1,2))
        epistemic_uncertainty = predictive_uncertainty - aleatoric_uncertainty
        return {'pred': pred,
                'predictive': predictive_uncertainty,
                'aleatoric': aleatoric_uncertainty,
                'epistemic': epistemic_uncertainty}


# -------------------- METHOD RETURNING EXPLANATIONS --------------------------  

    def get_rel_vect(self, win_size, overlap=True):
        """
        Main method to use, will return a relevance vector.
        Input:  win_size    the window size (k in alg. 1)
                overlap     whether the windows should be overlapping, default is True
        Output: rel_vects   the relevance vectors, dimensions are:
                            - number of features (input)
                            - number of outputs (usually output layer, can be hidden layer)
                            to interpret the result, look at one output (e.g., the predicted class)
                            and visualise the input features in some way
        """

        # create array for relevance vectors, each element has dimensions (num_feats)*blobdimension
        # where the relevance of each feature on the different activations in that blob is stored
        rel_vects = {key: np.zeros((self.num_feats,
                                    self.temp_tar_val[key].shape[0]),
                                   dtype=np.float64) for key in
                     self.temp_tar_val}

        # a counts vector to keep track of how often a feature is marginalised out
        counts = np.zeros((self.num_feats), dtype=np.int)

        # a matrix where each entry reflects the index in the flattened input (image)
        all_feats = np.reshape([i for i in range(self.num_feats*self.channels)], self.x.shape)

        if overlap:
            row_ind = range(self.x.shape[1]-win_size+1)
            col_ind = range(self.x.shape[2]-win_size+1)
        else:
            row_ind = range(self.x.shape[1]/win_size)
            col_ind = range(self.x.shape[2]/win_size)


        for i in row_ind:
            start_time = time.time()
            for j in col_ind:
                # get the window which we want to simulate as unknown
                window = all_feats[:,i:i+win_size,j:j+win_size].ravel()
                # evaluate the prediction difference
                pred_diffs = self._get_rel_vect_subset(window)
                for key in pred_diffs:
                    rel_vects[key][window[window<self.num_feats]] \
                            += pred_diffs[key]
                counts[window[window<self.num_feats]] += 1
            print ("row {}/{} took: --- {:.4f} seconds --- ".format(i,self.x.shape[1]-win_size+1,(time.time() - start_time)))

        # get average relevance of each feature
        for key in rel_vects:
            rel_vects[key][counts!=0] = (rel_vects[key][counts!=0]/counts[counts!=0][:,np.newaxis] ).astype(np.float16)

        return rel_vects, counts


#%%        
# -------------------------- SAMPLING METHODS ---------------------------------


    def _get_rel_vect_subset(self, feature_set):
        """
        Returns the relevance vector, given the features that are unknown.
        Input:  notGivenFeats   indices of the raveled (!) feature vector
                                that are unknown
        """

        assert np.ndim(feature_set) == 1, 'expected a single feature set'

        # for each data point in X, replace the (not-given-)feature with the value for it seen in X
        x_new = np.zeros((self.num_samples,len(self.x.ravel())))
        x_new[:] = np.copy(self.x).ravel()[np.newaxis]

        x_new[:, feature_set.ravel()] = self.sampler.get_samples(feature_set, self.x, self.num_samples)

        # get prediction for the altered x-values
        x_new = x_new.reshape(self.num_samples, self.x.shape[0],
                              self.x.shape[1], self.x.shape[2])

        # append the original x (so that they use the same weight samples)
        all_x = np.concatenate((np.expand_dims(self.x, axis=0), x_new))

        # # uncommenting this block will save images with a patch marginalised
        # self.n_images += 1
        # f, ax = plt.subplots()
        # x_avg = np.mean(x_new, axis=0)
        # ax.imshow( np.transpose(x_avg, (1, 2, 0)), interpolation='nearest')
        # f.savefig('./patches/patch_{}.png'.format(self.n_images))
        # with open('./patches/patch_{}.p'.format(self.n_images), 'wb') as f:
        #     pickle.dump(x_avg, f)

        # all_x has dimensions num_samples x K x C
        # where each of num_samples applies the same dropout mask
        pred_mc = self.tar_func(all_x)

        # evaluate the prediction difference
        rel_vect = self._evaluate_salience(pred_mc)

        return rel_vect

# -------------- EVALUATION OF THE PREDICTION DIFFERENCE ----------------------        

    def _evaluate_salience(self, tarVals):
        '''
        Evaluating the prediction difference using the weight of evidence as
        well as salience maps on each type of uncertainty
        Input:  tarVals     the values of all the blobs for several altered inputs
                            length is self.num_blobs, dimensions of each blob
                            are (num_featsures)*(shape of blob)
        '''
        # average over all predictions received by using altered input values
        # we will save a value per feature map (instead of for each activation)
        # therefore we loop at the avg over all activations, and the max
        prediction_diffs = {}
        trainsize = self.sampler.n

        pred_original = np.expand_dims(tarVals[0], axis=0)
        pred_samples = tarVals[1:]

        self.true_tar_val = self.compute_uncertainties(pred_original)
        self.true_tar_val = {key:self.true_tar_val[key][0]
                             if np.ndim(self.true_tar_val[key]) > 1
                             else self.true_tar_val[key]
                             for key in self.true_tar_val}


        # integrate across samples from sampler
        prob_not_feat_mc = np.mean(pred_samples, axis=0, keepdims=True)

        result = self.compute_uncertainties(prob_not_feat_mc)
        # unravel to a 1-dim vector
        result['pred'] = result['pred'].ravel()

        # predictive difference on class probabilities
        # check that the sum to 1 (probabilities)
        assert abs(sum(self.true_tar_val['pred']) - 1) < 1e-6,\
                'invalid prob dist'
        assert abs(sum(result['pred']) - 1) < 1e-6,\
                'invalid prob dist'
        # do a laplace correction to avoid problems with zero probabilities
        tarVal_laplace = (self.true_tar_val['pred']*trainsize+1) \
                   / (trainsize+len(self.true_tar_val['pred']))
        avgP_laplace = (result['pred']*trainsize+1) \
                   / (trainsize+len(self.true_tar_val['pred']))
        # calculate the odds for the true targets and  the targets with some features marginalised out
        oddsTarVal = np.log2(tarVal_laplace/(1-tarVal_laplace))
        oddsAvgP = np.log2(avgP_laplace/(1-avgP_laplace))
        # take average over feature maps
        pd = oddsTarVal-oddsAvgP
        prediction_diffs['pred'] = pd

        for key in result:
            if key == 'pred':
                continue
            prediction_diffs[key] = self.true_tar_val[key] - result[key]

        return prediction_diffs

