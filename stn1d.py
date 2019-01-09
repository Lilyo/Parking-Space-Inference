import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def _meshgrid(height, width):
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])), tf.transpose(tf.expand_dims(tf.linspace(1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])), tf.transpose(tf.expand_dims(tf.linspace(1.0, 1.0, width), 1), [1, 0]))
    z_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])), tf.transpose(tf.expand_dims(tf.linspace(1.0, 1.0, width), 1), [1, 0]))

    grid = tf.concat([x_t, y_t], 0)
    grid = tf.concat([grid, z_t], 0)
    return grid

def _transform(theta, x, out_size):
    num_batch = tf.shape(x)[0]
    theta_r = tf.reshape(theta, (-1, 3, 3))
    theta_r = tf.cast(theta_r, 'float32')

    out_height = out_size[0]
    out_width = out_size[1]

    grid1 = _meshgrid(out_height, out_width)
    grid2 = tf.expand_dims(grid1, 0)
    grid3 = tf.reshape(grid2, [-1])
    x_flat = tf.reshape(tf.transpose(tf.squeeze(x), [0, 2, 1]), [-1])
    grid4 = tf.tile(grid3, tf.stack([num_batch]))
    x_grid4 = tf.multiply(x_flat, grid4)
    grid5 = tf.reshape(x_grid4, tf.stack([num_batch, 3, -1]))

    # Transform A x (x_t, 1)^T -> (x_s)
    T_g = tf.matmul(theta_r, grid5)
    T_g = tf.transpose(T_g, [0,2,1])
    T_g = tf.expand_dims(T_g, 1)
    return T_g, theta_r


