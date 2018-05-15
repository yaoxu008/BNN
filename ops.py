import tensorflow as tf

image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, output_dim, kernel_h, kernel_w, d_h=1, d_w=1, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel_h, kernel_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def mul_conv2d(input, prefix, output_dims, kernel_h, kernel_w, d_h=1, d_w=1, stddev=0.02, name="mul_conv2d"):
    with tf.variable_scope(name):
        size = len(output_dims)
        output = input
        for i in range(size):
            k_h = kernel_h[i]
            k_w = kernel_w[i]
            output_dim = output_dims[i]
            w = tf.get_variable(prefix + 'w_' + str(i), [k_h, k_w, input.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            output = tf.nn.conv2d(input, w, strides=[1, d_h, d_w, 1], padding='VALID')

            biases = tf.get_variable(prefix + 'biases_' + str(i), [output_dims[i]],
                                     initializer=tf.constant_initializer(0.0))

            output = tf.reshape(tf.nn.bias_add(output, biases), output.get_shape())
            bn = batch_norm(name=prefix + 'bn' + str(i))
            output = relu(bn(output))
        return output


def relu(x):
    return tf.nn.relu(x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.max_pool(
        x, ksize=ksize, strides=strides, padding=padding
    )


def flat(x):
    x_shape = x.get_shape().as_list()

    nodes = x_shape[1] * x_shape[2] * x_shape[3]

    return tf.reshape(x, [x_shape[0], nodes])


def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)


def Euclidean(x, y):
    temp = tf.square(x - y)
    # temp = tf.abs(x-y)
    temp = tf.reshape(tf.reduce_mean(temp, 1), [-1, 1])
    temp = tf.sqrt(temp)

    return temp


def Lossfunction1(x1, x2, flags, alpha, NP_ratio, batch_size):
    # batch_Flag = load.flagGen(flags, NP_ratio, alpha, batch_size)
    loss = tf.reduce_mean(flags * Euclidean(x1, x2)) + 1 / alpha

    return loss


def Lossfunction2(x1, x2, flags, alpha, NP_ratio, batch_size):
    difference = Euclidean(x1, x2)
    '''
    loss_ = sigmoid_cross_entropy_with_logits(difference, flags * -1 + 1)
    factor = load.flagGen(flags, alpha)
    loss = tf.reduce_mean(loss_ * factor)
    '''
    flags_ = (flags * -1 + 1)
    loss_ = tf.square(difference - flags_)
    loss = (tf.reduce_sum(loss_ * flags) / tf.reduce_sum(flags) + alpha * tf.reduce_sum(loss_ * flags_) / tf.reduce_sum(
        flags_)) / (1 + alpha)
    # loss = tf.reduce_mean(loss_)
    # factor = load.flagGen(flags, alpha)
    # loss = tf.reduce_mean(loss_ * factor)
    return loss
