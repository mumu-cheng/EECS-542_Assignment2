import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys
import scipy.io


def load_vgg_model(model_path, model_url):
    filename = model_url.split("/")[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        raise IOError("VGG model file not found! Please download it first!")
    data = scipy.io.loadmat(filepath)
    return data

def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)

def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2):
	if output_shape is None: # upsample by 2 by default
	    output_shape = x.get_shape().as_list()
	    output_shape[1] *= 2
	    output_shape[2] *= 2
	    output_shape[3] = W.get_shape().as_list()[2]
	conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
	return tf.nn.bias_add(conv, b)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")