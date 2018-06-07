# Uncertainty Estimation in Bayesian Neural Networks and Links to Interpretability

MLSALT thesis project
** currently just a private repository ** 

## Summary

TODO

## Setup
- add this directory to PYTHONPATH
- set up virtual environment using `requirements.txt`
- set up for gpu usage
	- add cuda paths to .bash_profile: needs libcublas.so.9.0 and libcudnn.so.7
		- e.g. `export LD_LIBRARY_PATH=~/cuda-9.0/lib64/:~/cuda/lib64`
	- for it to work on jupyter do these steps:
		1. check that sys.executable points to the same executable in ipython shell and jupyter ([see here](https://github.com/jupyter/notebook/issues/2120))
		2. add appropriate LD_LIBRARY_PATH to jupyter config ([see here](https://github.com/jupyter/notebook/issues/1290))
- to start jupyter without the gpu, use `CUDA_VISIBLE_DEVICES="" jupyter notebook`	


