import os
from definitions import ROOT_DIR
import numpy as np
from keras.utils import np_utils
import keras.datasets


def load_mnist(flatten, channels_first=False, val_frac=0.1):
    """
    Loads MNIST data
    MNIST dataset cached in ~/.keras/datasets
    If flatten=True, then channels_first argument is ignored
    """
    (x_train, y_train), (x_test, y_test) = \
            keras.datasets.mnist.load_data(path='mnist.npz')

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train / 255
    x_test = x_test / 255

    # create validation partition
    np.random.seed(0)
    n = x_train.shape[0]
    nb_val = int(val_frac * n)
    indices = np.random.permutation(n)
    val_idx, training_idx = indices[:nb_val], indices[nb_val:]
    x_train, x_val = x_train[training_idx,:], x_train[val_idx,:]
    y_train, y_val = y_train[training_idx,:], y_train[val_idx,:]

    # reshaping 
    if flatten:
        x_train = np.reshape(x_train, (-1, 784))
        x_val = np.reshape(x_val, (-1, 784))
        x_test = np.reshape(x_test, (-1, 784))
    else:
        if channels_first:
            x_train = np.expand_dims(x_train, axis=1)
            x_val = np.expand_dims(x_val, axis=1)
            x_test = np.expand_dims(x_test, axis=1)
        else:
            x_train = np.expand_dims(x_train, axis=-1)
            x_val = np.expand_dims(x_val, axis=-1)
            x_test = np.expand_dims(x_test, axis=-1)

    return ((x_train, y_train),
            (x_val, y_val),
            (x_test, y_test))


def load_cifar10(channels_first=False, val_frac=0.1):
    """
    Loads cifar10 data
    Cifar10 dataset cached in ~/.keras/datasets
    """
    (x_train, y_train), (x_test, y_test) = \
            keras.datasets.cifar10.load_data()

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train / 255
    x_test = x_test / 255

    # create validation partition
    np.random.seed(0)
    n = x_train.shape[0]
    nb_val = int(val_frac * n)
    indices = np.random.permutation(n)
    val_idx, training_idx = indices[:nb_val], indices[nb_val:]
    x_train, x_val = x_train[training_idx,:], x_train[val_idx,:]
    y_train, y_val = y_train[training_idx,:], y_train[val_idx,:]

    # reshaping 
    if channels_first:
        x_train = np.transpose(x_train, axes=(0, 3, 1, 2))
        x_val = np.transpose(x_val, axes=(0, 3, 1, 2))
        x_test = np.transpose(x_test, axes=(0, 3, 1, 2))

    return ((x_train, y_train),
            (x_val, y_val),
            (x_test, y_test))

def load_svhn(channels_first=False, val_frac=0.1):
    """
    Loads SVHN data
    SVHN dataset cached in ROOT_DIR/datasets
    ROOT_DIR is defined in definitions.py as project home directory
    """

    urls = ['http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
            'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
            'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat']
    directory = os.path.join(ROOT_DIR, 'datasets', 'SVHN')
    os.makedirs(directory, exist_ok=True)

    if not os.path.isfile(os.path.join(directory, 'train_32x32.mat')):
        print("Downloading train_32x32.mat...")
        os.system('wget -P {} "{}"'.format(directory, urls[0]))
    if not os.path.isfile(os.path.join(directory, 'test_32x32.mat')):
        print("Downloading test_32x32.mat...")
        os.system('wget -P {} "{}"'.format(directory, urls[1]))
    if not os.path.isfile(os.path.join(directory, 'extra_32x32.mat')):
        print("Downloading extra_32x32.mat...")
        os.system('wget -P {} "{}"'.format(directory, urls[2]))

    import scipy.io as sio
    train = sio.loadmat(os.path.join(directory, 'train_32x32.mat'))
    test = sio.loadmat(os.path.join(directory, 'test_32x32.mat'))

    x_train = train['X']
    x_test = test['X']
    y_train = train['y']
    y_test = test['y']

    # svhn assigns class label 10 to digit 0
    # reassign it to class label 0
    np.place(y_train, y_train == 10, 0)
    np.place(y_test, y_test == 10, 0)

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train / 255
    x_test = x_test / 255

    # reshaping 
    if channels_first:
        x_train = np.transpose(x_train, axes=(3, 2, 0, 1))
        x_test = np.transpose(x_test, axes=(3, 2, 0, 1))
    else:
        x_train = np.transpose(x_train, axes=(3, 0, 1, 2))
        x_test = np.transpose(x_test, axes=(3, 0, 1, 2))

    # create validation partition
    np.random.seed(0)
    n = x_train.shape[0]
    nb_val = int(val_frac * n)
    indices = np.random.permutation(n)
    val_idx, training_idx = indices[:nb_val], indices[nb_val:]
    x_train, x_val = x_train[training_idx,:], x_train[val_idx,:]
    y_train, y_val = y_train[training_idx,:], y_train[val_idx,:]

    return ((x_train, y_train),
            (x_val, y_val),
            (x_test, y_test))


