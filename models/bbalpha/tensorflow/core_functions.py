import tensorflow as tf

def logsumexp(x, axis=None):
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    return tf.log(tf.reduce_sum(tf.exp(x - x_max), axis=axis,
                                keepdims=True)) + x_max


def bbalpha_softmax_cross_entropy_with_mc_logits(mc_logits, y_dup, alpha):
    mc_log_softmax = mc_logits - tf.reduce_max(mc_logits, axis=2,
                                               keepdims=True)
    mc_log_softmax = mc_log_softmax - \
            tf.log(tf.reduce_sum(tf.exp(mc_log_softmax), axis=2,
                                 keepdims=True))
    mc_ll = tf.reduce_sum(tf.to_float(y_dup) *
                          mc_log_softmax, -1)  # N x K

    K_mc = tf.cast(tf.shape(mc_ll)[1], tf.float32)

    # this is the loss function
    # reduce_mean instead of reduce_sum eliminates dependence on batch size
    return - 1. / alpha * tf.reduce_mean((logsumexp(alpha * mc_ll, 1) + \
                                          tf.log(1.0 / K_mc)))

def nn_layer(input_tensor, layer_name, dropout, act=tf.nn.relu):
    """
    Reusable code for making a simple neural net layer: 
    matrix multiply, bias add, activation
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        # uses keep_prob rather than dropout
        dropped = tf.nn.dropout(input_tensor, 1-dropout) # handles scaling
        weights = tf.get_variable('weight_variable')
        biases = tf.get_variable('bias_variable')
        preactivate = tf.matmul(dropped, weights) + biases
        activations = act(preactivate, name='activation')
        return activations

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],
                     padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1],
                          strides=[1, k, k, 1], padding='VALID')
