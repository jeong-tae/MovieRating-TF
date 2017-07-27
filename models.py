import numpy as np
import tensorflow as tf

def linear_fn(x, K = 20, stddev = 0., name = None):
    initializer = tf.random_normal_initializer(stddev = stddev)
    _shape = tf.cast(x, tf.float32).get_shape().as_list()
    if len(_shape) > 2:
        raise ValueError
    w = tf.Variable(initializer([_shape[1], K]), name = name)
    return tf.matmul(x, w)

def bilinear(user, item):

    s = linear_fn(user, K = 20, stddev = 0.02, name = 'U')
    s_b = linear_fn(user, K = 1, stddev = 0.02, name = 'u')
    t = linear_fn(item, K = 20, stddev = 0.02, name = 'V')
    t_b = linear_fn(item, K = 1, stddev =0.02, name = 'v')

    b = s_b + t_b

    _r = tf.reduce_sum(s * t, reduction_indices = 1)
    _r = tf.reshape(_r, [-1, 1])

    _r = _r + b

    return _r


