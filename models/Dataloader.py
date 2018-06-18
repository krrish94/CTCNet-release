"""
@author: Ganesh Iyer, Krishna Murthy
"""

import numpy as np
import os
import scipy.misc as smc
from skimage.util import random_noise
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt


import liefunctions as lie


class Dataloader:

	def __init__(self, config):

		# Config module
		self.config = __import__(config)

		# Batch size
		self.batch_size = self.config.net_hyperparams['batch_size']
		# Sequence length (used only if it is an LSTM)
		self.seq_len = self.config.net_hyperparams['sequence_length']

		# Number of frames to load per partition
		self.partition_limit = self.config.dataloader_params['partition_limit']
		# Number of train, val frames
		self.total_frames_train = self.config.dataloader_params['total_frames_train']
		self.total_frames_validation = self.config.dataloader_params['total_frames_validation']

		# Input image dims, mean, stddev
		self.IMG_HT = self.config.input_params['IMG_HT']
		self.IMG_WD = self.config.input_params['IMG_WD']
		self.norm_mean = self.config.input_params['model_input_mean']
		self.norm_std = self.config.input_params['model_input_std']

		# Paths to train and test metadata (.txt files)
		self.train_set = np.loadtxt(self.config.paths['TRAINING_PARSER_PATH'], dtype = str)
		self.validation_set = np.loadtxt(self.config.paths['TESTING_PARSER_PATH'], dtype = str)
		self.base_path = self.config.paths['DATASET_PATH']


	# Shuffle the train set (we don't have to shuffle the validation set)
	def shuffle(self):
		np.random.shuffle(self.train_set)


	# Shuffle the train set for the (non-CTC) LSTM (we don't have to shuffle the validation set)
	# This shuffle works even for CTC variants
	def shuffle_lstm(self):
		np.random.shuffle(self.train_set.reshape(self.train_set.shape[0] / self.seq_len, self.seq_len, 18))


	# Load a partition from the train set.
	# If the number of samples in the metadata file is less that those that are requested,
	# return a False flag.
	# Arguments:
	# p_no: partition number (index of the partition to be loaded)
	# isTrainPhase: True when training, False during validation
	def loadpartition_cnn(self, p_no, isTrainPhase = True, addNoiseToImage = False, \
		addNoiseToLabel = False, centerCrop = False):

		# By default assume that this method will succeed in loading the requested number of samples
		success = True

		# Get the appropriate partition from the dataset
		if isTrainPhase:
			dataset_part = self.train_set[p_no*self.partition_limit:(p_no+1)*self.partition_limit]
			# If there are insufficient samples, set the 'success' flag to False
			if (p_no + 1) * self.partition_limit > self.total_frames_train:
				success = False
		else:
			dataset_part = self.validation_set[p_no*self.partition_limit:(p_no+1)*self.partition_limit]
			if (p_no + 1) * self.partition_limit > self.total_frames_validation:
				success = False

		# Each line from the text file has the form 'img1_path', 'img2_path', 'transform'
		frame_one_list = dataset_part[:,0]
		frame_two_list = dataset_part[:,1]
		transforms = np.float32(dataset_part[:,2:])

		# Create variables to store the input images and the corresponding label (se(3)
		# exponential coordinates)
		input_container = np.zeros((self.partition_limit, self.IMG_HT, self.IMG_WD, 6), dtype = np.float32)
		xi_container = np.zeros((self.partition_limit, 6), dtype = np.float32)

		if success:
			# Load data from the current partition
			c_idx = 0
			for f_one_name, f_two_name, transform in tqdm(zip(frame_one_list, frame_two_list, transforms)):

				# Load the first image
				source_img = smc.imread(os.path.join(self.base_path, f_one_name))
				if not centerCrop:
					source_img = np.float32(resize(source_img, (self.IMG_HT, self.IMG_WD), preserve_range=True))
				else:
					# Center crop the image (to square)
					widthDiff = source_img.shape[1] - source_img.shape[0]
					startWidthInd = int(np.floor(widthDiff / 2.0))
					source_img = source_img[:, startWidthInd:startWidthInd+source_img.shape[0]]
					source_img = np.float32(resize(source_img, (self.IMG_HT, self.IMG_WD), \
						preserve_range = True))
				source_img = (source_img) / 255.0

				# Load the second image
				target_img = smc.imread(os.path.join(self.base_path, f_two_name))
				if not centerCrop:
					target_img = np.float32(resize(target_img, (self.IMG_HT, self.IMG_WD), preserve_range=True))
				else:
					# Center crop the image (to square)
					widthDiff = target_img.shape[1] - target_img.shape[0]
					startWidthInd = int(np.floor(widthDiff / 2.0))
					target_img = target_img[:, startWidthInd:startWidthInd+target_img.shape[0]]
					target_img = np.float32(resize(target_img, (self.IMG_HT, self.IMG_WD), \
						preserve_range = True))
				target_img = (target_img) / 255.0

				# Randomly (with prob ~ 0.5) add slight Gaussian noise to the images.
				# Note that the variance parameter of the Gaussian noise is set to 0.0002 empirically.
				# This works well for 7scenes. Not sure if this generalizes to other datasets.
				if addNoiseToImage:
					coinFlip = np.random.uniform()
					if coinFlip > 0.5:
						source_img = random_noise(source_img, mode = 'gaussian', var = 0.0002)
						target_img = random_noise(target_img, mode = 'gaussian', var = 0.0002)

				# Normalize the images
				source_img = (source_img -  self.norm_mean) / self.norm_std
				target_img = (target_img -  self.norm_mean) / self.norm_std

				# Append the current image pair, and corresponding se(3) exp coordinates to the partition
				input_container[c_idx, :, :, :] = np.concatenate((source_img, target_img), 2)
				xi_container[c_idx, :] = lie.SE3_logmap(transform.reshape(4,4))
				# Optionally add some noise to the se(3) exponential coordinates
				# This is useful when you're not training on Ground-Truth data and you're instead
				# training from data generated by a 'noisy teacher', such as an existing SLAM framework.
				if addNoiseToLabel:
					xi_container[c_idx, :] = xi_container[c_idx, :] + \
					np.concatenate((0.0001 * np.random.randn(1,3), 0.00005 * np.random.randn(1,3)),1)

				c_idx += 1

		# Return variables in the format the network expects (dimensions need to be adjusted)
		input_container = input_container.reshape(self.partition_limit / self.batch_size, self.batch_size, \
			self.IMG_HT, self.IMG_WD , 6)
		xi_container = xi_container.reshape(self.partition_limit / self.batch_size, self.batch_size, 6)

		return input_container, xi_container, success

	def loadpartition_long_seq(self, p_no, isTrainPhase = True, addNoiseToImage = False, \
		addNoiseToLabel = False, centerCrop = True):

		# By default assume that this method will succeed in loading the requested number of samples
		success = True

		# Get the appropriate partition from the dataset
		if isTrainPhase:
			chunk_part = self.train_set[p_no*(self.partition_limit/self.seq_len):(p_no + 1)*(self.partition_limit/self.seq_len)]
			# dataset_part = self.train_set[p_no*self.partition_limit:(p_no+1)*self.partition_limit]
			# If there are insufficient samples, set the 'success' flag to False
			if (p_no + 1) * self.partition_limit > self.total_frames_train:
				success = False
		else:
			# dataset_part = self.validation_set[p_no*self.partition_limit:(p_no+1)*self.partition_limit]
			chunk_part = self.validation_set[p_no*(self.partition_limit/self.seq_len):(p_no+1)*(self.partition_limit/self.seq_len)]
			if (p_no + 1) * self.partition_limit > self.total_frames_validation:
				success = False

		# Each line from the text file has the form 'img1_path', 'img2_path', 'transform'


		input_container = np.zeros((self.partition_limit, self.IMG_HT, self.IMG_WD, 6), dtype = np.float32)

		xi_container_skip_zero = np.zeros((self.partition_limit, 6), dtype = np.float32)
		xi_container_skip_one = np.zeros((self.partition_limit/2, 6), dtype = np.float32)
		xi_container_skip_two = np.zeros((self.partition_limit/3, 6), dtype = np.float32)

		# no_windows = self.partition_limit/self.seq_len

		if success:
			# Load data from the current partition
			c_idx = 0
			f_idx = 0
			for chunk in tqdm(chunk_part):

				skip_0_set = np.loadtxt("./cache/lstm_transforms_files/" + chunk[0] + "/dense_file_1.txt", dtype = str)[int(int(chunk[1])):int(chunk[1]) + self.seq_len]
				skip_1_set = np.loadtxt("./cache/lstm_transforms_files/" + chunk[0] + "/dense_file_2.txt", dtype = str)[int(chunk[1]):int(chunk[1]) + self.seq_len:2]
				skip_2_set = np.loadtxt("./cache/lstm_transforms_files/" + chunk[0] + "/dense_file_3.txt", dtype = str)[int(chunk[1]):int(chunk[1]) + self.seq_len:3]

				#Dealing with frames first

				frame_one_list = skip_0_set[:,0]
				frame_two_list = skip_0_set[:,1]
				transforms_zero_list = np.float32(skip_0_set[:,2:])
				transforms_one_list = np.float32(skip_1_set[:,2:])
				transforms_two_list = np.float32(skip_2_set[:,2:])

				for f_one_name, f_two_name in zip(frame_one_list, frame_two_list):
					# print (f_one_name, f_two_name)
					source_img = smc.imread(os.path.join(self.base_path, f_one_name))
					if not centerCrop:
						source_img = np.float32(resize(source_img, (self.IMG_HT, self.IMG_WD), preserve_range=True))
					else:
						# Center crop the image (to square)
						widthDiff = source_img.shape[1] - source_img.shape[0]
						startWidthInd = int(np.floor(widthDiff / 2.0))
						source_img = source_img[:, startWidthInd:startWidthInd+source_img.shape[0]]
						source_img = np.float32(resize(source_img, (self.IMG_HT, self.IMG_WD), \
							preserve_range = True))
					source_img = (source_img) / 255.0

					# Load the second image
					target_img = smc.imread(os.path.join(self.base_path, f_two_name))
					if not centerCrop:
						target_img = np.float32(resize(target_img, (self.IMG_HT, self.IMG_WD), preserve_range=True))
					else:
						# Center crop the image (to square)
						widthDiff = target_img.shape[1] - target_img.shape[0]
						startWidthInd = int(np.floor(widthDiff / 2.0))
						target_img = target_img[:, startWidthInd:startWidthInd+target_img.shape[0]]
						target_img = np.float32(resize(target_img, (self.IMG_HT, self.IMG_WD), \
							preserve_range = True))
					target_img = (target_img) / 255.0

					# Randomly (with prob ~ 0.5) add slight Gaussian noise to the images.
					# Note that the variance parameter of the Gaussian noise is set to 0.0002 empirically.
					# This works well for 7scenes. Not sure if this generalizes to other datasets.
					if addNoiseToImage:
						coinFlip = np.random.uniform()
						if coinFlip > 0.5:
							source_img = random_noise(source_img, mode = 'gaussian', var = 0.0002)
							target_img = random_noise(target_img, mode = 'gaussian', var = 0.0002)

					# Normalize the images
					source_img = (source_img -  self.norm_mean) / self.norm_std
					target_img = (target_img -  self.norm_mean) / self.norm_std

					input_container[f_idx, :, :, :] = np.concatenate((source_img, target_img), 2)
					# xi_container[c_idx, :] = lie.SE3_logmap(transform.reshape(4,4))

					f_idx+=1

				# print c_idx

				for idx in range(transforms_zero_list.shape[0]):
					xi_container_skip_zero[c_idx*18 + idx] = lie.SE3_logmap(transforms_zero_list[idx].reshape(4,4))

				for idx in range(transforms_one_list.shape[0]):
					xi_container_skip_one[c_idx*9 + idx] = lie.SE3_logmap(transforms_one_list[idx].reshape(4,4))

				for idx in range(transforms_two_list.shape[0]):
					xi_container_skip_two[c_idx*6 + idx] = lie.SE3_logmap(transforms_two_list[idx].reshape(4,4))

				c_idx += 1


			input_container = input_container.reshape(self.partition_limit / self.seq_len, self.seq_len, \
				self.IMG_HT, self.IMG_WD , 6)
			xi_container_skip_zero = xi_container_skip_zero.reshape(self.partition_limit / self.seq_len, self.seq_len, 6)
			xi_container_skip_one = xi_container_skip_one.reshape(self.partition_limit / self.seq_len, self.seq_len/2, 6)
			xi_container_skip_two = xi_container_skip_two.reshape(self.partition_limit / self.seq_len, self.seq_len/3, 6)


			xi_container = np.concatenate([xi_container_skip_zero, xi_container_skip_one, xi_container_skip_two], axis=1)

			# print xi_container[1]



			return input_container, xi_container, success
