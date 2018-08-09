# Training models

- bbalpha
	- models with dropout variational distribution and alpha-divergence loss function
	- the original implementation was in keras, but here its also implemented equivalently in tensorflow and pytorch
- bbalpha-train-n
	- scripts to train models with bb-alpha object by subsampling the trainin data
- deterministic
	- DNN model with equivalent architecture.

Code for the BB-alpha dropout model in keras is modified from [here](https://github.com/YingzhenLi/Dropout_BBalpha)


Credit goes the original authors:
Li, Y., & Gal, Y. (2017). Dropout inference in bayesian neural networks with alpha-divergences. arXiv preprint arXiv:1703.02914.

Modifications here:
- add metrics in keras to save log-likelihood and avg-acc over training
- change bbalpha loss function to include alpha=0
- add function to load a saved trained model, and generate a test model with a different number of MC samples
