from keras.layers import Input, Dense, Activation, Flatten,\
        Conv2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.regularizers import l2


def build_cnn(inp, p, nb_units, nb_classes, wd):
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)
    conv1 = Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                   padding='valid', kernel_regularizer=l2(wd),
                   activation='relu')(inp)
    conv2 = Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                   kernel_regularizer=l2(wd), activation='relu')(conv1)
    pool = MaxPooling2D(pool_size=pool_size)(conv2)
    flat = Flatten()(pool)
    dropout1 = Dropout(p)(flat)
    dense1 = Dense(nb_units, kernel_regularizer=l2(wd),
                   activation='relu')(dropout1)
    dropout2 = Dropout(p)(dense1)
    dense2 = Dense(nb_classes,kernel_regularizer=l2(wd),
                   activation='softmax')(dropout2)
    model = Model(inputs=inp, outputs=dense2)
    return model
