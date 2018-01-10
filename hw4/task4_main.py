#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from task4_data import generate_data

tf.reset_default_graph()  # for iPython convenience

# how to define recurrent layers in tensorflow:

cell_type = 'simple'
#cell_type = 'gru'
#cell_type = 'lstm'

# define recurrent layer
if cell_type == 'simple':
  cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)
elif cell_type == 'lstm':
  cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
elif cell_type == 'gru':
  cell = tf.nn.rnn_cell.GRUCell(num_hidden)
else:
  raise ValueError('bad cell type.')

# wrap this layer in a recurrent neural network
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# notes:
# - this tensor is unrolled in time (it contains the hidden states of all time
#   points)
# - the recurrent weights are encapsuled, we do not need to define them
# - you should only use the outputs (not the hidden states) for creating the
#   output neuron

# get the unit outputs at the last time step
last_outputs = outputs[:,-1,:]

# use on top of this a sigmoid neuron for sequence classification

# ...



