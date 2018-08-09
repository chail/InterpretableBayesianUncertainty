# Uncertainty Estimation in Bayesian Neural Networks and Links to Interpretability

MLSALT thesis project

## Summary

Bayesian neural networks provide model predictions as well as uncertainty in the prediction. But what features of the input make the prediction uncertain? This project investigates this by seeing how predictive, epistemic, and aleatoric uncertainty change when pixels of an image are known versus unknown.

## Roadmap
- models/
	- neural network implementations
- salience/
	- code for generating pixel-based visualisations of uncertainty contributions
- experiments/
	- ipython notebooks for doing BNN validation experiments, and plotting visualisations
	- (make sure to generate some salience visualisations above before running expt 3)
- toolbox/
	- core functions for loading data and computing uncertainty

## Setup Steps
- add this directory to PYTHONPATH
- set up virtual environment using `requirements.txt`
- to set up for gpu usage
	- add cuda paths to .bash_profile: needs libcublas.so.9.0 and libcudnn.so.7
		- e.g. `export LD_LIBRARY_PATH=~/cuda-9.0/lib64/:~/cuda/lib64`
	- for it to work on jupyter do these steps:
		1. check that sys.executable points to the same executable in ipython shell and jupyter ([see here](https://github.com/jupyter/notebook/issues/2120))
		2. add appropriate LD_LIBRARY_PATH to jupyter config ([see here](https://github.com/jupyter/notebook/issues/1290))
- to start jupyter without the gpu, use `CUDA_VISIBLE_DEVICES="" jupyter notebook`	


