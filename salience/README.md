# Visualising uncertainty

This code computes the change in uncertainty when pixels are known vs. unknown to generate visualisations.

After training models in `../models/bbalpha/keras/` you can change the settings in `experiments.py` to generate visualisations.


## Credits 
Code modified from [Predictive Difference Analysis](https://github.com/lmzintgraf/DeepVis-PredDiff)

Credit goes the original authors:
Zintgraf, L. M., Cohen, T. S., Adel, T., & Welling, M. (2017). Visualizing deep neural network decisions: Prediction difference analysis. arXiv preprint arXiv:1702.04595.

Modifications here:
- Compute difference in predictive, epistemic, and aleatoric uncertainties based on a Bayesian neural network
- Use the same draw from the variational weight distribution over all samples when marginalising pixels
- This means that the network should evaluate only samples over the same patch at each time, so the computation logic is changed
- Modify classifier utilities to handle a BNN
- Modify visualisation pipeline to show all uncertainties

## How to modify to use your own models
1) Train your model on MNIST, CIFAR10, or SVHN
2) In `utils_classifiers.py` implement an object that loads a saved model and performs a forward pass. The current implementation is for a Keras model
3) Update the configurations in `experiments.py` starting from line 35.
	- There are 2 modes: for DNN and BNN. DNN computes predictive difference only. BNN also computes 3 uncertainties.
	- For DNN modes, the output of a forward pass should have dimensions N x C, and for a BNN N x K x C. N is the number of input images, K is the number of weight samples, and C is the number of classes.
