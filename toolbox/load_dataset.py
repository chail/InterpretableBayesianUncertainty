import os
from definitions import ROOT_DIR
import numpy as np
from keras.utils import np_utils
import keras.datasets
from timeit import default_timer as timer



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

def load_cifar100(channels_first=False, val_frac=0.1):
    """
    Loads cifar100 data
    Cifar100 dataset cached in ~/.keras/datasets
    """
    (x_train, y_train), (x_test, y_test) = \
            keras.datasets.cifar100.load_data(label_mode='coarse')

    y_train = np_utils.to_categorical(y_train, 20)
    y_test = np_utils.to_categorical(y_test, 20)
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


def load_image_data(dataset, channels_first=False):
    '''
    load train, val, test images and labels of the dataset
    Input:  dataset         string of either 'mnist', 'cifar10', 'cifar100',
                             'svhn', 'isic'
            channels_first  if true, shape of returned images is
                            (n, channels, height, width),
                            otherwise (n, height, width, channels)
    Returns:
        tuple of (train, val, test) splits containing image data and one-hot
        labels
            images are the first element of each split, e.g. train[0]
            labels are the second element of each split, e.g. train[1]
    '''
    if dataset == 'mnist':
        return load_mnist(False, channels_first)
    elif dataset == 'cifar10':
        return load_cifar10(channels_first)
    elif dataset == 'cifar100':
        return load_cifar100(channels_first)
    elif dataset == 'svhn':
        return load_svhn(channels_first)
    elif dataset == 'isic':
        return load_isic(channels_first)
    else:
        return None


def load_image_labels(dataset):
    '''
    Input:  dataset     string of either 'mnist', 'cifar10', 'cifar100',
                        'svhn', 'isic'
    Returns:
        list of label names for the provided dataset
    '''
    if dataset == 'mnist':
        return [str(x) for x in range(10)]
    elif dataset == 'cifar10':
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset == 'cifar100':
        return ['aquatic mammals', 'fish', 'flowers', 'food containers',
                'fruit and vegetables', 'household electrical devices',
                'household furniture', 'insects', 'large carnivores',
                'large man-made outdoor things',
                'large natural outdoor scenes',
                'large omnivores and herbivores',
                'medium-sized mammals', 'non-insect invertebrates',
                'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1',
                'vehicles 2']
    elif dataset == 'svhn':
        return [str(x) for x in range(10)]
    elif dataset == 'isic':
        return ['MEL', 'NV', 'BCC', 'BKL']
    else:
        return None



def load_cats(channels_first=False):

    from skimage import io
    directory = os.path.join(ROOT_DIR, 'datasets', 'cats', 'resized')
    cats = [io.imread(os.path.join(directory, f))
            for f in os.listdir(directory)]
    cats = np.stack(cats, axis=0)
    print(cats.shape)

    cats = cats / 255
    print(np.max(cats))
    print(np.min(cats))

    if channels_first:
        cats = np.transpose(cats, axes=(0, 3, 1, 2))

    n = cats.shape[0]
    y = np.zeros((n, 10))
    y[:, 3] = 1 # cat label in cifar 10

    return cats, y


def load_isic(channels_first=False, ntest=200, nval=200, sample='undersample'):

    from skimage import io
    from collections import Counter
    from imblearn.over_sampling import RandomOverSampler

    classes_dir = os.path.join(ROOT_DIR, 'datasets', 'isic2018', 'classes')
    img_dir = os.path.join(ROOT_DIR, 'datasets', 'isic2018',
                           'resize')

    classes = ['MEL', 'NV', 'BCC', 'BKL']

    x_data = []
    y_data = []

    # load images
    for ii, c in enumerate(classes):
        start = timer()
        with open(os.path.join(classes_dir, c + '.txt'), 'r') as f:
            lines = f.readlines()
            img_data = [io.imread(os.path.join(img_dir, l.strip() + '.jpg'))
                        for l in lines]
            img_data = np.stack(img_data, axis=0)
            x_data.append(img_data)
            labels = np.zeros((img_data.shape[0], len(classes)))
            labels[:, ii] = 1
            y_data.append(labels)
        end = timer()
        print("Loaded {} in {} s".format(c, end-start))

    # reshuffle before splitting into train/test/val
    np.random.seed(0)
    for ii in range(len(classes)):
        assert x_data[ii].shape[0] == y_data[ii].shape[0]
        n = x_data[ii].shape[0]
        perm = np.random.permutation(n)
        x_data[ii] = x_data[ii][perm]
        y_data[ii] = y_data[ii][perm]
        if channels_first:
            x_data[ii] = np.transpose(x_data[ii], axes=(0, 3, 1, 2))

        print(x_data[ii].shape)
        print(y_data[ii].shape)


    ntest_class = int(ntest / len(classes))
    nval_class = int(nval / len(classes))
    min_examples = min(len(x) for x in x_data)
    assert ntest_class + nval_class < min_examples, \
            "decrease test and val partition sizes"

    # split test and val partitions
    x_test = np.concatenate([x[:ntest_class] for x in x_data], axis=0)
    y_test = np.concatenate([y[:ntest_class] for y in y_data], axis=0)
    x_val = np.concatenate([x[ntest_class:ntest_class+nval_class]
                            for x in x_data], axis=0)
    y_val = np.concatenate([y[ntest_class:ntest_class+nval_class]
                            for y in y_data], axis=0)

    # randomly resample underrepresented classes for train partitions
    if sample == 'oversample':
        ros = RandomOverSampler(random_state=0)
        x_remaining = np.concatenate([x[ntest_class+nval_class:]
                                      for x in x_data], axis=0)
        y_remaining = np.concatenate([y[ntest_class+nval_class:]
                                      for y in y_data], axis=0)
        (n, d1, d2, d3) = x_remaining.shape
        x_remaining = np.reshape(x_remaining, (n, d1 * d2 * d3))
        print("Counts before resampling")
        print(sorted(Counter(np.argmax(y_remaining, axis=1)).items()))
        x_train, y_train = ros.fit_sample(x_remaining,
                                          np.argmax(y_remaining, axis=1))
        print("Counts after resampling")
        print(sorted(Counter(y_train).items()))
        x_train = np.reshape(x_train, (-1, d1, d2, d3))
        y_train = np_utils.to_categorical(y_train, len(classes))
    elif sample == 'undersample':
        x_remaining = np.concatenate([x[ntest_class+nval_class:min_examples]
                                      for x in x_data], axis=0)
        y_remaining = np.concatenate([y[ntest_class+nval_class:min_examples]
                                      for y in y_data], axis=0)
        print("Counts with undersampling")
        print(sorted(Counter(np.argmax(y_remaining, axis=1)).items()))
        x_train = x_remaining
        y_train = y_remaining
    else:
        raise ValueError("sample should be 'undersample' or 'oversample'")

    # reshuffle order within each train/val/test split
    perm = np.random.permutation(x_train.shape[0])
    x_train = x_train[perm]
    y_train = y_train[perm]
    perm = np.random.permutation(x_val.shape[0])
    x_val = x_val[perm]
    y_val = y_val[perm]
    perm = np.random.permutation(x_test.shape[0])
    x_test = x_test[perm]
    y_test = y_test[perm]

    x_train = x_train / 255
    x_val = x_val / 255
    x_test = x_test / 255

    print(np.min(x_train))
    print(np.max(x_train))

    return ((x_train, y_train),
            (x_val, y_val),
            (x_test, y_test))
