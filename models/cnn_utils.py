"""
@author Ganesh Iyer, J. Krishna Murthy
"""

import cPickle
import numpy as np
import tensorflow as tf


# Create a weight variable initialized with a normal distribution
# (truncated to two standard devs)
# shape: list (denotes the size of the weight variable)
# layerno: an integer to be added to the name of the weight variable returned
# decay: a multiplier that decays each weight (eg. decay of 0.1 returns 0.1 * W as opposed to W)
def weight_variable(shape, layerno, decay = 1.0):

	W = decay * tf.get_variable("weight_%d"%layerno, shape = shape, initializer = tf.contrib.layers.xavier_initializer())
	return W


# Create a (constant) bias vector of specified shape
# shape: list (denotes the size of the bias variable)
# layerno: an integer to be added to the name of the bias variable returned
# constant: the constant value which each entry of the bias vector is to be initialized to
def bias_variable(shape, layerno, constant = 0.0):

	B = tf.Variable(tf.constant(constant, shape = shape, dtype=tf.float32), name = "bias_%d"%layerno)
	return B


# Initialize weights of a particular layer
# W: weight tensor
# layerno: an integer denoting the name of the layer to be initialized with the given weight tensor
# isTrainable: boolean value; indicates whether the layer is to be trained
# layerType: one of 'weight' or 'bias'
def init_weights(W, layerType, layerno, isTrainable):

	w_init = tf.constant_initializer(W)
	weight = tf.get_variable(layerType + '_%d'%layerno, shape = W.shape, dtype = tf.float32, initializer = w_init, trainable = isTrainable)
	return weight



def conv2d_batchnorm(x, W, name, phase, beta_r, gamma_r, mean_r, variance_r, bias = None, scale = True):

	beta = tf.constant_initializer(beta_r)
	gamma = tf.constant_initializer(gamma_r)
	moving_mean = tf.constant_initializer(mean_r)
	moving_variance = tf.constant_initializer(variance_r)

	with tf.name_scope(name):
		temp1 = tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME') + bias
		with tf.name_scope('batch_norm'):
			temp2 = tf.contrib.layers.batch_norm(temp1, param_initializers = {'beta': beta, 'gamma': gamma, 'moving_mean': moving_mean, 'moving_variance': moving_variance}, is_training = phase, updates_collections = None, trainable = True, scale = scale)
			return tf.nn.relu(temp2, 'relu')


def conv2d_batchnorm_init(x, W, name, phase, stride, padding):

	with tf.name_scope(name):
		temp1 = tf.nn.conv2d(x, W, strides = stride, padding = padding)
		temp2 = tf.nn.relu(temp1, 'relu')
		with tf.name_scope('batch_norm'):
			return tf.contrib.layers.batch_norm(temp2, is_training = phase, updates_collections = None, trainable = True)


def conv2d_batchnorm_noscale(x, W, name, phase, beta_r, mean_r, variance_r, bias = None, stride = [1,1,1,1]):

	beta = tf.constant_initializer(beta_r)
	moving_mean = tf.constant_initializer(mean_r)
	moving_variance = tf.constant_initializer(variance_r)

	with tf.name_scope(name):
		if bias:
			temp1 = tf.nn.conv2d(x, W, strides = stride, padding = 'SAME') + bias
		else:
			temp1 = tf.nn.conv2d(x, W, strides = stride, padding = 'SAME')
		temp2 = tf.nn.relu(temp1, 'relu')
		with tf.name_scope('batch_norm'):
			bn_out = tf.contrib.layers.batch_norm(temp2, param_initializers = {'beta': beta, 'moving_mean': moving_mean, 'moving_variance': moving_variance}, is_training = phase, updates_collections = None, trainable = True, scale = False)
			return bn_out


def conv2d_init(x, W, name, stride, padding):

	with tf.name_scope(name):
		temp1 = tf.nn.conv2d(x, W, strides = stride, padding = padding)
		temp2 = tf.nn.relu(temp1, 'relu')
		return temp2


def conv2d_bias_init(x, W, b, name):

	with tf.name_scope(name):
		temp1 = tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME') + b
		temp2 = tf.nn.relu(temp1, 'relu')
		return temp2


def max_pool(x, kernel, stride, name):
	return tf.nn.max_pool(x, ksize = kernel, strides = stride, padding = 'SAME', name = name)


# Attach a lot of summaries to a tensor (for TensorBoard visualization)
def variable_summaries(var):

	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		sum_mean = tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		sum_stddev = tf.summary.scalar('stddev', stddev)
		sum_hist = tf.summary.histogram('histogram', var)
		return [sum_mean, sum_hist, sum_stddev]


def load_weights(vgg11_pretrained_weights_path):
	with open(vgg11_pretrained_weights_path, 'rb') as f:
		parameters = cPickle.load(f)

	convIndices = [0, 4, 8, 11, 15, 18, 22, 25]
	weights = []
	biases = []

	beta = []
	gamma = []
	running_means = []
	running_vars = []

	for idx in convIndices:
		W_conv = parameters['features.' + str(idx) + '.weight']
		W_conv = np.swapaxes(W_conv, 0, 3)
		W_conv = np.swapaxes(W_conv, 1, 2)
		b_conv = parameters['features.' + str(idx) + '.bias']

		batch_norm_w = parameters['features.' + str(idx + 1) + '.weight']
		batch_norm_b = parameters['features.' + str(idx + 1) + '.bias']
		batch_norm_mean = parameters['features.' + str(idx + 1) + '.running_mean']
		batch_norm_var = parameters['features.' + str(idx + 1) + '.running_var']

		weights.append(W_conv)
		biases.append(b_conv)
		beta.append(batch_norm_w)
		gamma.append(batch_norm_b)
		running_means.append(batch_norm_mean)
		running_vars.append(batch_norm_var)

	return weights, biases, beta, gamma, running_means, running_vars
