Code based on https://github.com/lmzintgraf/DeepVis-PredDiff with heavy modifications.

Credit goes the original authors:
Zintgraf, L. M., Cohen, T. S., Adel, T., & Welling, M. (2017). Visualizing deep neural network decisions: Prediction difference analysis. arXiv preprint arXiv:1702.04595.

Modifications in this version:
- Compute difference in predictive, epistemic, and aleatoric uncertainties based on a Bayesian neural network
- Use the same draw from the variational weight distribution over all samples when marginalising pixels
- This means that the network should evaluate only samples over the same patch at each time (change the computation logic)
- Modify classifier utilities to handle a BNN
- Modify visualisation pipeline to show all uncertainties

