"""
@author Ganesh Iyer, Krishna Murthy
Specifies various architectures tried out
"""

import os
import tensorflow as tf


from cnn_utils import *


# Extract VGG11_BN weights
def extractWeights_VGG11_BN(vgg11_pretrained_weights_path, centerCrop = False):

	# Read in weights, biases, BN params from .npy text file into numpy arrays
	all_weights, all_biases, betas, gammas, means, variances = load_weights(vgg11_pretrained_weights_path)

	# Initialize TF variables to hold weights, biases, BN params, and transfer values
	# from the numpy arrays to their corresponding TF counterparts
	vgg_train = True

	# We're stacking 2 input images along the channel dimension and passing them as
	# input to the network. So we duplicate the weights of the first conv layer
	# in the base VGG11_BN model
	W_conv1_1 = init_weights(all_weights[0], 'weight', 100, vgg_train)
	W_conv1_2 = init_weights(all_weights[0], 'weight', 101, vgg_train)

	W_conv1 = tf.concat([W_conv1_1, W_conv1_2], 2)
	b_conv1 = init_weights(all_biases[0], 'bias', 0, vgg_train)

	W_conv2 = init_weights(all_weights[1], 'weight', 1, vgg_train)
	b_conv2 = init_weights(all_biases[1], 'bias', 1, vgg_train)

	W_conv3 = init_weights(all_weights[2], 'weight', 2, vgg_train)
	b_conv3 = init_weights(all_biases[2], 'bias', 2, vgg_train)

	W_conv4 = init_weights(all_weights[3], 'weight', 3, vgg_train)
	b_conv4 = init_weights(all_biases[3], 'bias', 3, vgg_train)

	W_conv5 = init_weights(all_weights[4], 'weight', 4, vgg_train)
	b_conv5 = init_weights(all_biases[4], 'bias', 4, vgg_train)

	W_conv6 = init_weights(all_weights[5], 'weight', 5, vgg_train)
	b_conv6 = init_weights(all_biases[5], 'bias', 5, vgg_train)

	W_conv7 = init_weights(all_weights[6], 'weight', 6, vgg_train)
	b_conv7 = init_weights(all_biases[6], 'bias', 6, vgg_train)

	W_conv8 = init_weights(all_weights[7], 'weight', 7, vgg_train)
	b_conv8 = init_weights(all_biases[7], 'bias', 7, vgg_train)

	W_ext1 = weight_variable([3,3,512,256], 8)
	W_ext2 = weight_variable([3,3,256,128], 9)
	W_ext3 = weight_variable([1,1,128,128], 10)
	W_ext4 = weight_variable([1,1,128,64], 11)
	if not centerCrop:
		W_fc1 = weight_variable([1280,6], 12, 0.001)
	else:
		W_fc1 = weight_variable([1024,6], 12, 0.001)

	model_weights = [W_conv1, W_conv2, W_conv3, W_conv4, W_conv5, W_conv6, W_conv7, W_conv8, \
		W_ext1, W_ext2, W_ext3, W_ext4, W_fc1]
	model_bias = [b_conv1, b_conv2, b_conv3, b_conv4, b_conv5, b_conv6, b_conv7, b_conv8]

	# Attach TF summaries to weights and biases
	summaries = []
	for weight_idx in range(len(model_weights)):
		summaries += variable_summaries(model_weights[weight_idx])
	for bias_idx in range(len(model_bias)):
		summaries += variable_summaries(model_bias[bias_idx])

	return model_weights, model_bias, betas, gammas, means, variances, summaries


# Extract VGG11_BN weights (pretrained weights for 2-view VO estimation)
def extractWeights_VGG11_trained(vgg11_pretrained_weights_path, retrainCNN = False):

	# Read in weights, biases, BN params from .npy text file into numpy arrays
	pre_weights = np.load(os.path.join(vgg11_pretrained_weights_path, 'model_weights.npy'))[()]
	pre_biases = np.load(os.path.join(vgg11_pretrained_weights_path, 'model_biases.npy'))[()]
	pre_batchnorm = np.load(os.path.join(vgg11_pretrained_weights_path, 'model_batchnorm.npy'))[()]

	print('Pretrained CNN weights loaded')

	# Initialize TF variables to hold weights, biases, BN params, and transfer values
	# from the numpy arrays to their corresponding TF counterparts

	# We're stacking 2 input images along the channel dimension and passing them as
	# input to the network. So we duplicate the weights of the first conv layer
	# in the base VGG11_BN model
	W_conv1_1 = init_weights(pre_weights['weight_100'], 'weight', 100, isTrainable = retrainCNN)
	W_conv1_2 = init_weights(pre_weights['weight_101'], 'weight', 101, isTrainable = retrainCNN)

	W_conv1 = tf.concat([W_conv1_1, W_conv1_2], 2)
	b_conv1 = init_weights(pre_biases['bias_0'], 'bias', 0, isTrainable = retrainCNN)

	W_conv2 = init_weights(pre_weights['weight_1'], 'weight', 1, isTrainable = retrainCNN)
	b_conv2 = init_weights(pre_biases['bias_1'], 'bias', 1, isTrainable = retrainCNN)

	W_conv3 = init_weights(pre_weights['weight_2'], 'weight', 2, isTrainable = retrainCNN)
	b_conv3 = init_weights(pre_biases['bias_2'], 'bias', 2, isTrainable = retrainCNN)

	W_conv4 = init_weights(pre_weights['weight_3'], 'weight', 3, isTrainable = retrainCNN)
	b_conv4 = init_weights(pre_biases['bias_3'], 'bias', 3, isTrainable = retrainCNN)

	W_conv5 = init_weights(pre_weights['weight_4'], 'weight', 4, isTrainable = retrainCNN)
	b_conv5 = init_weights(pre_biases['bias_4'], 'bias', 4, isTrainable = retrainCNN)

	W_conv6 = init_weights(pre_weights['weight_5'], 'weight', 5, isTrainable = retrainCNN)
	b_conv6 = init_weights(pre_biases['bias_5'], 'bias', 5, isTrainable = retrainCNN)

	W_conv7 = init_weights(pre_weights['weight_6'], 'weight', 6, isTrainable = retrainCNN)
	b_conv7 = init_weights(pre_biases['bias_6'], 'bias', 6, isTrainable = retrainCNN)

	W_conv8 = init_weights(pre_weights['weight_7'], 'weight', 7, isTrainable = retrainCNN)
	b_conv8 = init_weights(pre_biases['bias_7'], 'bias', 7, isTrainable = retrainCNN)

	W_ext1 = init_weights(pre_weights['weight_8'], 'weight', 8, isTrainable = retrainCNN)
	W_ext2 = init_weights(pre_weights['weight_9'], 'weight', 9, isTrainable = retrainCNN)
	W_ext3 = init_weights(pre_weights['weight_10'], 'weight', 10, isTrainable = retrainCNN)
	W_ext4 = init_weights(pre_weights['weight_11'], 'weight', 11, isTrainable = retrainCNN)
	#fc scale term added
	W_fc1 = 0.001*init_weights(pre_weights['weight_12'], 'weight', 12, isTrainable = retrainCNN)

	model_weights = [W_conv1, W_conv2, W_conv3, W_conv4, W_conv5, W_conv6, W_conv7, W_conv8, \
		W_ext1, W_ext2, W_ext3, W_ext4, W_fc1]
	model_bias = [b_conv1, b_conv2, b_conv3, b_conv4, b_conv5, b_conv6, b_conv7, b_conv8]

	betas = [pre_batchnorm['BatchNorm/beta'], pre_batchnorm['BatchNorm_1/beta'], \
		pre_batchnorm['BatchNorm_2/beta'], pre_batchnorm['BatchNorm_3/beta'], \
		pre_batchnorm['BatchNorm_4/beta'], pre_batchnorm['BatchNorm_5/beta'], \
		pre_batchnorm['BatchNorm_6/beta'], pre_batchnorm['BatchNorm_7/beta'], \
		pre_batchnorm['BatchNorm_8/beta'], pre_batchnorm['BatchNorm_9/beta'], \
		pre_batchnorm['BatchNorm_10/beta'], pre_batchnorm['BatchNorm_11/beta']]

	gammas = [pre_batchnorm['BatchNorm/gamma'], pre_batchnorm['BatchNorm_1/gamma'], \
		pre_batchnorm['BatchNorm_2/gamma'], pre_batchnorm['BatchNorm_3/gamma'], \
		pre_batchnorm['BatchNorm_5/gamma'], pre_batchnorm['BatchNorm_5/gamma'], \
		pre_batchnorm['BatchNorm_6/gamma'], pre_batchnorm['BatchNorm_7/gamma']]

	means = [pre_batchnorm['BatchNorm/moving_mean'], pre_batchnorm['BatchNorm_1/moving_mean'], \
		pre_batchnorm['BatchNorm_2/moving_mean'], pre_batchnorm['BatchNorm_3/moving_mean'], \
		pre_batchnorm['BatchNorm_4/moving_mean'], pre_batchnorm['BatchNorm_5/moving_mean'], \
		pre_batchnorm['BatchNorm_6/moving_mean'], pre_batchnorm['BatchNorm_7/moving_mean'], \
		pre_batchnorm['BatchNorm_8/moving_mean'], pre_batchnorm['BatchNorm_9/moving_mean'], \
		pre_batchnorm['BatchNorm_10/moving_mean'], pre_batchnorm['BatchNorm_11/moving_mean']]

	variances = [pre_batchnorm['BatchNorm/moving_variance'], pre_batchnorm['BatchNorm_1/moving_variance'], \
		pre_batchnorm['BatchNorm_2/moving_variance'], pre_batchnorm['BatchNorm_3/moving_variance'], \
		pre_batchnorm['BatchNorm_4/moving_variance'], pre_batchnorm['BatchNorm_5/moving_variance'], \
		pre_batchnorm['BatchNorm_6/moving_variance'], pre_batchnorm['BatchNorm_7/moving_variance'], \
		pre_batchnorm['BatchNorm_8/moving_variance'], pre_batchnorm['BatchNorm_9/moving_variance'], \
		pre_batchnorm['BatchNorm_10/moving_variance'], pre_batchnorm['BatchNorm_11/moving_variance']]

	# Attach TF summaries to weights and biases
	summaries = []
	for weight_idx in range(len(model_weights)):
		summaries += variable_summaries(model_weights[weight_idx])
	for bias_idx in range(len(model_bias)):
		summaries += variable_summaries(model_bias[bias_idx])

	return model_weights, model_bias, betas, gammas, means, variances, summaries


"""
CNN model (based on VGG11-BN)
"""
class CNN_VGG11_BN:

	def __init__(self, input_x, isTrain, keep_prob, batch_size, vgg11_pretrained_weights_path, \
		isPartOfLSTM = False, retrainCNN = True, end2end = False, centerCrop = False):

		self.input_x = input_x
		self.isTrain = isTrain
		self.keep_prob = keep_prob
		self.batch_size = batch_size

		self.isPartOfLSTM = isPartOfLSTM
		self.retrainCNN = retrainCNN
		self.end2end = end2end

		self.centerCrop = centerCrop

		if self.isPartOfLSTM and not end2end:
			# In this case vgg11_pretrained_weights_path contains path to the dir containing weights
			# for the pretrained CNN for relative pose estimation
			[self.weights, self.biases, self.bn_betas, self.bn_gammas, self.bn_mov_means, \
				self.bn_mov_vars, self.summaries] = extractWeights_VGG11_trained(\
					vgg11_pretrained_weights_path, retrainCNN = self.retrainCNN)
		else:
			[self.weights, self.biases, self.bn_betas, self.bn_gammas, self.bn_mov_means, \
				self.bn_mov_vars, self.summaries] = extractWeights_VGG11_BN(vgg11_pretrained_weights_path, \
					centerCrop = self.centerCrop)


	def inference(self):

		layer0 = conv2d_batchnorm(self.input_x, self.weights[0], 'conv_0', self.isTrain, self.bn_betas[0], \
			self.bn_gammas[0], self.bn_mov_means[0], self.bn_mov_vars[0], bias = self.biases[0])
		layer0_m = max_pool(layer0, [1,2,2,1], [1,2,2,1], 'max_pool0')

		layer1 = conv2d_batchnorm(layer0_m, self.weights[1], 'conv_1', self.isTrain, self.bn_betas[1], \
			self.bn_gammas[1], self.bn_mov_means[1], self.bn_mov_vars[1], bias = self.biases[1])
		layer1_m = max_pool(layer1, [1,2,2,1], [1,2,2,1], 'max_pool1')

		layer2 = conv2d_batchnorm(layer1_m, self.weights[2], 'conv_2', self.isTrain, self.bn_betas[2], \
			self.bn_gammas[2], self.bn_mov_means[2], self.bn_mov_vars[2], bias = self.biases[2])
		layer3 = conv2d_batchnorm(layer2, self.weights[3], 'conv_3', self.isTrain, self.bn_betas[3], \
			self.bn_gammas[3], self.bn_mov_means[3], self.bn_mov_vars[3], bias = self.biases[3])
		layer3_m = max_pool(layer3, [1,2,2,1], [1,2,2,1], 'max_pool3')

		layer4 = conv2d_batchnorm(layer3_m, self.weights[4], 'conv_2', self.isTrain, self.bn_betas[4], \
			self.bn_gammas[4], self.bn_mov_means[4], self.bn_mov_vars[4], bias = self.biases[4])
		layer5 = conv2d_batchnorm(layer4, self.weights[5], 'conv_5', self.isTrain, self.bn_betas[5], \
			self.bn_gammas[5], self.bn_mov_means[5], self.bn_mov_vars[5], bias = self.biases[5])
		layer5_m = max_pool(layer5, [1,2,2,1], [1,2,2,1], 'max_pool5')

		layer6 = conv2d_batchnorm(layer5_m, self.weights[6], 'conv_6', self.isTrain, self.bn_betas[6], \
			self.bn_gammas[6], self.bn_mov_means[6], self.bn_mov_vars[6], bias = self.biases[6])
		layer7 = conv2d_batchnorm(layer6, self.weights[7], 'conv_7', self.isTrain, self.bn_betas[7], \
			self.bn_gammas[7], self.bn_mov_means[7], self.bn_mov_vars[7], bias = self.biases[7])
		layer7_m = max_pool(layer7, [1,2,2,1], [1,2,2,1], 'max_pool7')

		if self.isPartOfLSTM and not self.end2end:
			layer8 = conv2d_batchnorm_noscale(layer7_m, self.weights[8], 'conv_8', self.isTrain, \
				self.bn_betas[8], self.bn_mov_means[8], self.bn_mov_vars[8], bias = None, stride = [1,2,2,1])
			layer9 = conv2d_batchnorm_noscale(layer8, self.weights[9], 'conv_9', self.isTrain, \
				self.bn_betas[9], self.bn_mov_means[9], self.bn_mov_vars[9], bias = None, stride = [1,1,1,1])
			layer10 = conv2d_batchnorm_noscale(layer9, self.weights[10], 'conv_10', self.isTrain, \
				self.bn_betas[10], self.bn_mov_means[10], self.bn_mov_vars[10], bias = None, stride = [1,1,1,1])
			layer11 = conv2d_batchnorm_noscale(layer10, self.weights[11], 'conv_11', self.isTrain, \
				self.bn_betas[11], self.bn_mov_means[11], self.bn_mov_vars[11], bias = None, stride = [1,1,1,1])
		else:
			layer8 = conv2d_batchnorm_init(layer7_m, self.weights[8], 'conv_8', self.isTrain, \
				stride = [1,2,2,1], padding = 'SAME')
			layer9 = conv2d_batchnorm_init(layer8, self.weights[9], 'conv_9', self.isTrain, \
				stride = [1,1,1,1], padding = 'SAME')
			layer10 = conv2d_batchnorm_init(layer9, self.weights[10], 'conv_10', self.isTrain, \
				stride = [1,1,1,1], padding = 'SAME')
			layer11 = conv2d_batchnorm_init(layer10, self.weights[11], 'conv_11', self.isTrain, \
				stride = [1,1,1,1], padding = 'SAME')
		
		if not self.centerCrop:
			layer11_m = tf.reshape(layer11, [self.batch_size, 1280])
		else:
			layer11_m = tf.reshape(layer11, [self.batch_size, 1024])

		# if self.isPartOfLSTM:
		# return layer11_m, self.summaries
		# else:
		layer11_drop = tf.nn.dropout(layer11_m, self.keep_prob)
		layer11_vec = (tf.matmul(layer11_drop, self.weights[12]))
		return layer11_vec, layer11_m, self.summaries
