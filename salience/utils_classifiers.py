# -*- coding: utf-8 -*-
"""

Utility methods for handling the classifiers:
Each class should implement:
    __init__():     load the model
    forward_pass(self, x):  evaluate a forward pass on x, output predictions
    with shape (N, K, C) where N is the number of examples in x, K is the
    number of samples from the weight distribution, C is the number of
    predictive classes

"""

import numpy as np
import keras
from keras.models import load_model
import os

class BBalpha_keras_net:
    def __init__(self, modelpath):
        '''
        Load trained bbalpha keras model
        Input:  modelpath   path to bbalpha test model
        Output: model
        '''
        self.test_model = load_model(modelpath)


    def forward_pass(self, x):
        '''
        Defines a forward pass of the keras model
        Input:  x    input data to the model
        Output: evaluated results for inputs (softmax MC samples)
        '''

        assert np.ndim(x) == 4, 'input shape (n, rows, cols, channels)'
        assert x.shape[-1] == 1 or x.shape[-1], \
                'number of channels == 1 or 3'
        out = self.test_model.predict(x)
        return out
