import numpy as np
import tensorflow as tf


def dynamic_RNN(x, weights, output_size, n_input, n_hidden, reuse, dropout):
	"""
	Dynamic RNN Cell:

	Input: window of vectors
	Output: Scale consistent (time window) vectors
	"""

	x = tf.reshape(x, [1, output_size, n_input])

	if reuse:
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, reuse = True)
	else:
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, reuse = False)

	lstm_cell = network = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=dropout)

	rnn_outputs, state = tf.nn.dynamic_rnn(lstm_cell, x, dtype = tf.float32)
	cell_op = rnn_outputs[0]

	cell_op_drop = tf.nn.dropout(cell_op, dropout)
	lstm_op = tf.matmul(cell_op_drop, weights)
	lstm_op = tf.reshape(lstm_op, [output_size, 6])

	return lstm_op
