import numpy as np
from numpy import ma

# -------------------- METHOD COMPUTING UNCERTAINTIES--------------------------  

def compute_uncertainties(pred_mc):
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
