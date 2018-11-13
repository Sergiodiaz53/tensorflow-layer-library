# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Some example architectures to be used with corresponding config files in folder configs/examples.

Overview:
---------
ArchitectureDense: Simple dense layer network (use with main_lstm.py)
ArchitectureLSTM: Most simple usage example of LSTM (use with main_lstm.py)
ArchitectureLSTM... : More advanced (optimized/flexible) usage examples of LSTM (use with main_lstm.py)
ArchitectureConvLSTM: Example for ConvLSTM and plotting (use with main_convlstm.py)

AutoencoderDynamicLenght: Dynamic lenght, LSTM autoenc. arjona@ml.jku.at
"""

import tensorflow as tf
import sys
import time
from collections import OrderedDict
import numpy as np
from TeLL.config import Config
from TeLL.initializations import constant, weight_xavier_conv2d
from TeLL.utility.misc_tensorflow import layers_from_specs, tensor_shape_with_flexible_dim
from TeLL.layers import ConcatLayer, ConvLSTMLayer, ConvLayer, DenseLayer, LSTMLayer, RNNInputLayer, MaxPoolingLayer, \
    ScalingLayer, LSTMLayerGetNetInput
from TeLL.utility.misc import get_rec_attr
from TeLL.regularization import regularize


class ArchitectureDense(object):
    def __init__(self, config: Config, dataset):
        """Simple network with dense layer and dense output layer;

        Command-line usage:
        >>> python3 samples/main_lstm.py --config=samples/config_dense.json

        Example input shapes: [n_samples, n_features]
        Example output shapes: [n_samples, n_features]
        """
        #
        # Some convenience objects
        #
        # We will use a list to store all layers for regularization etc. (this is optional)
        layers = []
        # Prepare xavier initialization for weights
        w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)

        #
        # Create placeholders for input data (shape: [n_samples, n_features])
        #
        input_shapes = dataset.get_input_shapes()
        X = tf.placeholder(tf.float32, shape=input_shapes['X'].shape)
        y_ = tf.placeholder(tf.float32, shape=input_shapes['y'].shape)
        n_output_units = dataset.datareader.get_num_classes()  # nr of output features is number of classes

        # ----------------------------------------------------------------------------------------------------------
        # Define network architecture
        # ----------------------------------------------------------------------------------------------------------

        #
        # Dense Layer
        #  Input for the dense layer shall be X (TeLL layers take tensors or TeLL Layer instances as input)
        #
        print("\tDense layer...")

        dense_layer = DenseLayer(incoming=X, n_units=config.n_dense, name='DenseLayer', W=w_init, b=tf.zeros,
                                 a=tf.nn.elu)
        layers.append(dense_layer)

        #
        # Output Layer
        #
        print("\tOutput layer...")
        output_layer = DenseLayer(incoming=dense_layer, n_units=n_output_units, name='DenseLayerOut', W=w_init,
                                  b=tf.zeros, a=tf.sigmoid)
        layers.append(output_layer)

        #
        # Calculate output
        #  This will calculate the output of output_layer, including all dependencies
        #
        output = output_layer.get_output()

        print("\tDone!")

        #
        # Publish
        #
        self.X = X
        self.y_ = y_
        self.output = output
        # Store layers in list for regularization in main file
        self.__layers = layers

    def get_layers(self):
        return self.__layers


class ArchitectureLSTM(object):
    def __init__(self, config: Config, dataset):
        """Simple network with LSTM layer and dense output layer; All sequence positions are fed to the LSTM layer at
        once, this is the most convenient but least flexible design; see ArchitectureLSTM_optimized for a faster
        version;

        Command-line usage:
        >>> python3 samples/main_lstm.py --config=samples/config_lstm.json

        Example input shapes: [n_samples, n_sequence_positions, n_features]
        Example output shapes: [n_samples, n_sequence_positions, n_features] (with return_states=True),
        [n_samples, 1, n_features] (with return_states=False)
        """
        #
        # Some convenience objects
        #
        # We will use a list to store all layers for regularization etc. (this is optional)
        layers = []
        # Prepare xavier initialization for weights
        w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)

        #
        # Create placeholders for input data (shape: [n_samples, n_sequence_positions, n_features])
        #
        input_shapes = dataset.get_input_shapes()
        X = tf.placeholder(tf.float32, shape=input_shapes['X'].shape)
        y_ = tf.placeholder(tf.float32, shape=input_shapes['y'].shape)
        n_output_units = dataset.datareader.get_num_classes()  # nr of output features is number of classes

        # ----------------------------------------------------------------------------------------------------------
        # Define network architecture
        # ----------------------------------------------------------------------------------------------------------

        #
        # LSTM Layer
        #  We want to create an output sequence with the LSTM instead of only returning the ouput at the last sequence
        #  position -> return_states=True
        #
        print("\tLSTM...")
        lstm_layer = LSTMLayer(incoming=X, n_units=config.n_lstm, name='LSTM',
                               W_ci=w_init, W_ig=w_init, W_og=w_init, W_fg=w_init,
                               b_ci=tf.zeros, b_ig=tf.zeros, b_og=tf.zeros, b_fg=tf.zeros,
                               a_ci=tf.tanh, a_ig=tf.sigmoid, a_og=tf.sigmoid, a_fg=tf.sigmoid, a_out=tf.nn.elu,
                               c_init=tf.zeros, h_init=tf.zeros, forgetgate=True, precomp_fwds=True, return_states=True)
        layers.append(lstm_layer)

        #
        # Output Layer
        #
        print("\tOutput layer...")
        output_layer = DenseLayer(incoming=lstm_layer, n_units=n_output_units, name='DenseLayerOut',
                                  W=w_init, b=tf.zeros, a=tf.sigmoid)
        layers.append(output_layer)

        #
        # Calculate output
        #
        output = output_layer.get_output(tickersteps=config.tickersteps)

        print("\tDone!")

        #
        # Publish
        #
        self.X = X
        self.y_ = y_
        self.output = output
        # Store layers in list for regularization in main file
        self.__layers = layers

    def get_layers(self):
        return self.__layers


class ArchitectureLSTMFlexible(object):
    def __init__(self, config: Config, dataset):
        """Architecture with LSTM layer followed by dense output layer; Inputs are fed to LSTM layer sequence position
        by sequence position in a for-loop; this is the most flexible design, as showed e.g. in ArchitectureLSTM3;

        Command-line usage:
        Change entry
        "architecture": "sample_architectures.ArchitectureLSTM"
        to
        "architecture": "sample_architectures.ArchitectureLSTMFlexible" in samples/config_lstm.json. Then run
        >>> python3 samples/main_lstm.py --config=samples/config_lstm.json

        Example input shapes: [n_samples, n_sequence_positions, n_features]
        Example output shapes: [n_samples, n_sequence_positions, n_features] (with return_states=True),
        [n_samples, 1, n_features] (with return_states=False)
        """
        #
        # Some convenience objects
        #
        # We will use a list to store all layers for regularization etc. (this is optional)
        layers = []
        # Prepare xavier initialization for weights
        w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)

        #
        # Create placeholders for input data (shape: [n_samples, n_sequence_positions, n_features])
        #
        X = tf.placeholder(tf.float32, shape=dataset.X_shape)
        y_ = tf.placeholder(tf.float32, shape=dataset.y_shape)
        n_output_units = dataset.y_shape[-1]  # nr of output features is number of classes
        n_seq_pos = dataset.X_shape[1]  # dataset.X_shape is [sample, seq_pos, features]

        # ----------------------------------------------------------------------------------------------------------
        # Define network architecture
        # ----------------------------------------------------------------------------------------------------------

        #
        # Input Layer
        #  RNNInputLayer will hold the input to network at each sequence position. We will initalize it with zeros-
        #  tensor of shape [sample, 1, features]
        #
        input_shape = dataset.X_shape[:1] + (1,) + dataset.X_shape[2:]
        rnn_input_layer = RNNInputLayer(tf.zeros(input_shape, dtype=tf.float32))
        layers.append(rnn_input_layer)

        #
        # LSTM Layer
        #
        print("\tLSTM...")
        lstm_layer = LSTMLayer(incoming=rnn_input_layer, n_units=config.n_lstm, name='LSTM',
                               W_ci=w_init, W_ig=w_init, W_og=w_init, W_fg=w_init,
                               b_ci=tf.zeros, b_ig=tf.zeros, b_og=tf.zeros, b_fg=tf.zeros,
                               a_ci=tf.tanh, a_ig=tf.sigmoid, a_og=tf.sigmoid, a_fg=tf.sigmoid, a_out=tf.nn.elu,
                               c_init=tf.zeros, h_init=tf.zeros, forgetgate=True, precomp_fwds=True, return_states=True)
        layers.append(lstm_layer)

        #
        # Output Layer
        #
        print("\tOutput layer...")
        output_layer = DenseLayer(incoming=lstm_layer, n_units=n_output_units, name='DenseLayerOut',
                                  W=w_init, b=tf.zeros, a=tf.sigmoid)
        layers.append(output_layer)

        # ----------------------------------------------------------------------------------------------------------
        # Loop through sequence positions and create graph
        # ----------------------------------------------------------------------------------------------------------

        #
        # Loop through sequence positions
        #
        print("\tRNN Loop...")
        for seq_pos in range(n_seq_pos):
            with tf.name_scope("Sequence_pos_{}".format(seq_pos)):
                print("\t  seq. pos. {}...".format(seq_pos))

                # Set rnn input layer to input at current sequence position
                rnn_input_layer.update(X[:, seq_pos:seq_pos + 1, :])

                # Calculate new network state at new frame (this updates the network's hidden activations, cell states,
                # and dependencies automatically)
                _ = lstm_layer.get_output()

        #
        # Loop through tickersteps
        #
        # Use zero input during ticker steps
        tickerstep_input = tf.zeros(dataset.X_shape[:1] + (1,) + dataset.X_shape[2:], dtype=tf.float32,
                                    name="tickerstep_input")

        for tickerstep in range(config.tickersteps):
            with tf.name_scope("Tickerstep_{}".format(tickerstep)):
                print("\t  tickerstep {}...".format(tickerstep))

                # Set rnn input layer to tickerstep input
                rnn_input_layer.update(tickerstep_input)

                # Calculate new network state at new frame (this updates the network's hidden activations, cell states,
                # and dependencies automatically)
                _ = lstm_layer.get_output(tickerstep_nodes=True)

        #
        # Calculate output but consider that the lstm_layer is already computed (i.e. do not modify cell states any
        # further)
        #
        output = output_layer.get_output(prev_layers=[lstm_layer])

        print("\tDone!")

        #
        # Publish
        #
        self.X = X
        self.y_ = y_
        self.output = output
        # Store layers in list for regularization in main file
        self.__layers = layers

    def get_layers(self):
        return self.__layers


# TODO: Implement continuous prediction
class ArchitectureLSTMOptimized(object):
    def __init__(self, config: Config, dataset):
        """Architecture with LSTM layer followed by dense output layer; Inputs are fed to LSTM layer sequence position
        by sequence position in tensorflow tf.while_loop();
        This is not as flexible as using a for-loop and more difficult to use but can be faster and optimized
        differently; LSTM return_states is not possible here unless manually implemented into the tf.while_loop (that is
        why we are only using the prediction at the last sequence position in this example);
        This is an advanced example, see ArchitectureLSTM to get started;

        Command-line usage:
        Change entry
        "architecture": "sample_architectures.ArchitectureLSTM"
        to
        "architecture": "sample_architectures.ArchitectureLSTMOptimized" in samples/config_lstm.json. Then run
        >>> python3 samples/main_lstm.py --config=samples/config_lstm.json

        Example input shapes: [n_samples, n_sequence_positions, n_features]
        Example output shapes: [n_samples, 1, n_features]
        """
        #
        # Some convenience objects
        #
        # We will use a list to store all layers for regularization etc. (this is optional)
        layers = []
        # Prepare xavier initialization for weights
        w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)

        #
        # Create placeholders for input data (shape: [n_samples, n_sequence_positions, n_features])
        #
        X = tf.placeholder(tf.float32, shape=dataset.X_shape)
        y_ = tf.placeholder(tf.float32, shape=dataset.y_shape)
        n_output_units = dataset.y_shape[-1]  # nr of output features is number of classes
        n_seq_pos = dataset.X_shape[1]  # dataset.X_shape is [sample, seq_pos, features]

        # ----------------------------------------------------------------------------------------------------------
        # Define network architecture
        # ----------------------------------------------------------------------------------------------------------

        #
        # Input Layer
        #  RNNInputLayer will hold the input to network at each sequence position. We will initalize it with zeros-
        #  tensor of shape [sample, 1, features]
        #
        input_shape = dataset.X_shape[:1] + (1,) + dataset.X_shape[2:]
        rnn_input_layer = RNNInputLayer(tf.zeros(input_shape, dtype=tf.float32))
        layers.append(rnn_input_layer)

        #
        # LSTM Layer
        #
        print("\tLSTM...")
        lstm_layer = LSTMLayer(incoming=rnn_input_layer, n_units=config.n_lstm, name='LSTM',
                               W_ci=w_init, W_ig=w_init, W_og=w_init, W_fg=w_init,
                               b_ci=tf.zeros, b_ig=tf.zeros, b_og=tf.zeros, b_fg=tf.zeros,
                               a_ci=tf.tanh, a_ig=tf.sigmoid, a_og=tf.sigmoid, a_fg=tf.sigmoid, a_out=tf.nn.elu,
                               c_init=tf.zeros, h_init=tf.zeros, forgetgate=True, precomp_fwds=True)
        layers.append(lstm_layer)

        #
        # Output Layer
        #
        print("\tOutput layer...")
        output_layer = DenseLayer(incoming=lstm_layer, n_units=n_output_units, name='DenseLayerOut',
                                  W=w_init, b=tf.zeros, a=tf.sigmoid)
        layers.append(output_layer)

        # ----------------------------------------------------------------------------------------------------------
        # Loop through sequence positions and create graph
        # ----------------------------------------------------------------------------------------------------------

        #
        # Loop through sequence positions
        #
        if n_seq_pos:
            def cond(seq_pos, *args):
                return seq_pos < n_seq_pos

            def body(seq_pos, lstm_h, lstm_c):
                # Set rnn input layer to input at current sequence position
                rnn_input_layer.update(X[:, seq_pos:seq_pos + 1, :])

                # Update lstm states
                lstm_layer.h[-1], lstm_layer.c[-1] = lstm_h, lstm_c

                # Calculate new network state at new frame (this updates the network's hidden activations, cell states,
                # and dependencies automatically)
                _ = lstm_layer.get_output()

                seq_pos = tf.add(seq_pos, 1)

                return seq_pos, lstm_layer.h[-1], lstm_layer.c[-1]

            with tf.name_scope("Sequence_pos"):
                print("\t  seq. pos. ...")
                wl_ret = tf.while_loop(cond=cond, body=body, loop_vars=(tf.constant(0),
                                                                        lstm_layer.h[-1], lstm_layer.c[-1]),
                                       parallel_iterations=10, back_prop=True, swap_memory=True)
                lstm_layer.h[-1], lstm_layer.c[-1] = wl_ret[-2], wl_ret[-1]

        #
        # Loop through tickersteps
        #
        if config.tickersteps:
            def cond(seq_pos, *args):
                return seq_pos < config.tickersteps

            def body(seq_pos, lstm_h, lstm_c):
                # Set rnn input layer to input at current sequence position
                rnn_input_layer.update(X[:, -1:, :])

                # Update lstm states
                lstm_layer.h[-1], lstm_layer.c[-1] = lstm_h, lstm_c

                # Calculate new network state at new frame (this updates the network's hidden activations, cell states,
                # and dependencies automatically)
                _ = lstm_layer.get_output(tickerstep_nodes=True)

                seq_pos = tf.add(seq_pos, 1)

                return seq_pos, lstm_layer.h[-1], lstm_layer.c[-1]

            with tf.name_scope("Tickersteps"):
                print("\t  tickersteps ...")
                wl_ret = tf.while_loop(cond=cond, body=body, loop_vars=(tf.constant(0),
                                                                        lstm_layer.h[-1], lstm_layer.c[-1]),
                                       parallel_iterations=10, back_prop=True, swap_memory=True)
                lstm_layer.h[-1], lstm_layer.c[-1] = wl_ret[-2], wl_ret[-1]

        #
        # Calculate output but consider that the lstm_layer is already computed
        #
        output = output_layer.get_output(prev_layers=[lstm_layer])

        print("\tDone!")

        #
        # Publish
        #
        self.X = X
        self.y_ = y_
        self.output = output
        # Store layers in list for regularization in main file
        self.__layers = layers

    def get_layers(self):
        return self.__layers


class ArchitectureLSTM3(object):
    def __init__(self, config: Config, dataset):
        """Architecture with LSTM layer followed by 2 dense layers and a dense output layer; The outputs of the 2 dense
        layers are used as additional recurrent connections for the LSTM; Inputs are fed to LSTM layer sequence
        position by sequence position in RNN loop; This is an advanced example, see ArchitectureLSTM to get started;

        Command-line usage:
        >>> python3 samples/main_lstm.py --config=samples/config_lstm3.json

        Example input shapes: [n_samples, n_sequence_positions, n_features]
        Example output shapes: [n_samples, n_sequence_positions, n_features] (with return_states=True),
        [n_samples, 1, n_features] (with return_states=False)
        """
        #
        # Some convenience objects
        #
        # We will use a list to store all layers for regularization etc. (this is optional)
        layers = []
        # Prepare xavier initialization for weights
        w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)

        #
        # Create placeholders for input data (shape: [n_samples, n_sequence_positions, n_features])
        #
        input_shapes = dataset.get_input_shapes()
        X = tf.placeholder(tf.float32, shape=input_shapes['X'].shape)
        y_ = tf.placeholder(tf.float32, shape=input_shapes['y'].shape)
        n_output_units = dataset.datareader.get_num_classes()  # nr of output features is number of classes

        n_seq_pos = input_shapes['X'].shape[1]  # dataset.X_shape is [sample, seq_pos, features]

        # ----------------------------------------------------------------------------------------------------------
        # Define network architecture
        # ----------------------------------------------------------------------------------------------------------

        #
        # Input Layer
        #  RNNInputLayer will hold the input to network at each sequence position. We will initalize it with zeros-
        #  tensor of shape [sample, 1, features]
        #
        input_shape = input_shapes['X'].shape[:1] + (1,) + input_shapes['X'].shape[2:]
        rnn_input_layer = RNNInputLayer(tf.zeros(input_shape, dtype=tf.float32))
        layers.append(rnn_input_layer)

        #
        # LSTM Layer
        #
        print("\tLSTM...")
        # We want to modify the number of recurrent connections -> we have to specify the shape of the recurrent weights
        rec_w_shape = (sum(config.n_dense_units) + config.n_lstm, config.n_lstm)
        # The forward weights can be initialized automatically, for the recurrent ones we will use our rec_w_shape
        lstm_w = [w_init, w_init(rec_w_shape)]
        lstm_layer = LSTMLayer(incoming=rnn_input_layer, n_units=config.n_lstm, name='LSTM',
                               W_ci=lstm_w, W_ig=lstm_w, W_og=lstm_w, W_fg=lstm_w,
                               b_ci=tf.zeros, b_ig=tf.zeros, b_og=tf.zeros, b_fg=tf.zeros,
                               a_ci=tf.tanh, a_ig=tf.sigmoid, a_og=tf.sigmoid, a_fg=tf.sigmoid, a_out=tf.nn.elu,
                               c_init=tf.zeros, h_init=tf.zeros, forgetgate=True, precomp_fwds=True, return_states=True)
        layers.append(lstm_layer)

        #
        # Dense Layers
        #
        print("\tDense layers...")
        dense_layers = list()
        for n_units in config.n_dense_units:
            dense_layers.append(DenseLayer(incoming=layers[-1], n_units=n_units, name='DenseLayer', W=w_init,
                                           b=tf.zeros, a=tf.nn.elu))
            layers.append(layers[-1])

        #
        # Use dense layers as additional recurrent input to LSTM
        #
        full_lstm_input = ConcatLayer([lstm_layer] + dense_layers, name='LSTMRecurrence')
        lstm_layer.add_external_recurrence(full_lstm_input)

        #
        # Output Layer
        #
        print("\tOutput layer...")
        output_layer = DenseLayer(incoming=dense_layers[-1], n_units=n_output_units, name='DenseLayerOut', W=w_init,
                                  b=tf.zeros, a=tf.sigmoid)
        layers.append(output_layer)

        # ----------------------------------------------------------------------------------------------------------
        # Loop through sequence positions and create graph
        # ----------------------------------------------------------------------------------------------------------

        #
        # Loop through sequence positions
        #
        print("\tRNN Loop...")
        for seq_pos in range(n_seq_pos):
            with tf.name_scope("Sequence_pos_{}".format(seq_pos)):
                print("\t  seq. pos. {}...".format(seq_pos))
                # Set rnn input layer to input at current sequence position
                layers[0].update(X[:, seq_pos:seq_pos + 1, :])

                # Calculate new lstm state (this automatically computes all dependencies, including rec. connections)
                _ = lstm_layer.get_output()

        #
        # Loop through tickersteps
        #
        # Use zeros as input during ticker steps
        tickerstep_input = tf.zeros(input_shape, dtype=tf.float32,
                                    name="tickerstep_input")

        for tickerstep in range(config.tickersteps):
            with tf.name_scope("Tickerstep_{}".format(tickerstep)):
                print("\t  tickerstep {}...".format(tickerstep))
                # Set rnn input layer to input at current sequence position
                layers[0].update(tickerstep_input)

                # Calculate new lstm state (this automatically computes all dependencies, including rec. connections)
                _ = lstm_layer.get_output(tickerstep_nodes=True)

        #
        # Calculate output but consider that the lstm_layer is already computed
        #
        output = output_layer.get_output(prev_layers=[lstm_layer])

        print("\tDone!")

        #
        # Publish
        #
        self.X = X
        self.y_ = y_
        self.output = output
        # Store layers in list for regularization in main file
        self.__layers = layers

    def get_layers(self):
        return self.__layers


class ArchitectureConvLSTM(object):
    def __init__(self, config: Config, dataset):
        """Example for convolutional network with convLSTM and convolutional output layer; Plots cell states, hidden
        states, X, y_, and a argmax over the convLSTM units outputs;

        Command-line usage:
        >>> python3 samples/main_convlstm.py --config=samples/config_convlstm.json

        Example input shapes: [n_samples, n_sequence_positions, x_dim, y_dim, n_features]
        Example output shapes: [n_samples, 1, x_dim, y_dim, n_features] (with return_states=True),
        [n_samples, 1, n_features] (with return_states=False)
        """
        #
        # Some convenience objects
        #
        # We will use a list to store all layers for regularization etc. (this is optional)
        layers = []
        # We will use xavier initialization later
        conv_W_initializer = weight_xavier_conv2d

        #
        # Create placeholders for feeding an input frame and a label at the each sequence position
        #
        X_shape = dataset.mb_info['X'][0]
        y_shape = dataset.mb_info['y'][0]
        n_classes = y_shape[-1]
        n_seq_pos = X_shape[1]  # dataset.X_shape is [sample, seq_pos, x, y, features)
        X = tf.placeholder(tf.float32, shape=X_shape)
        y_ = tf.placeholder(dataset.mb_info['y'][1], shape=y_shape)  # dataset.y_shape is [sample, seq_pos, features)

        # ----------------------------------------------------------------------------------------------------------
        # Define network architecture
        # ----------------------------------------------------------------------------------------------------------

        #
        # Initialize input to network of shape [sample, 1, x, y, features] with zero tensor of size of a frame
        #
        input_shape = X_shape[:1] + (1,) + X_shape[2:]
        rnn_input_layer = RNNInputLayer(tf.zeros(input_shape, dtype=tf.float32))
        layers.append(rnn_input_layer)

        #
        # ConvLSTM Layer
        #
        n_lstm = config.n_lstm  # number of output feature channels
        lstm_x_fwd = config.kernel_lstm_fwd  # x/y size of kernel for forward connections
        lstm_y_fwd = config.kernel_lstm_fwd  # x/y size of kernel for forward connections
        lstm_x_bwd = config.kernel_lstm_bwd  # x/y size of kernel for recurrent connections
        lstm_y_bwd = config.kernel_lstm_bwd  # x/y size of kernel for recurrent connections
        lstm_input_channels_fwd = rnn_input_layer.get_output_shape()[-1]  # number of input channels
        if config.reduced_rec_lstm:
            lstm_input_channels_bwd = config.reduced_rec_lstm  # number of recurrent connections (after squashing)
        else:
            lstm_input_channels_bwd = n_lstm  # number of recurrent connections

        # Here we create our kernels and biases for the convLSTM; See ConvLSTMLayer() documentation for more info;
        lstm_init = dict(W_ci=[conv_W_initializer([lstm_x_fwd, lstm_y_fwd, lstm_input_channels_fwd, n_lstm]),
                               conv_W_initializer([lstm_x_bwd, lstm_y_bwd, lstm_input_channels_bwd, n_lstm])],
                         W_ig=[conv_W_initializer([lstm_x_fwd, lstm_y_fwd, lstm_input_channels_fwd, n_lstm]),
                               conv_W_initializer([lstm_x_bwd, lstm_y_bwd, lstm_input_channels_bwd, n_lstm])],
                         W_og=[conv_W_initializer([lstm_x_fwd, lstm_y_fwd, lstm_input_channels_fwd, n_lstm]),
                               conv_W_initializer([lstm_x_bwd, lstm_y_bwd, lstm_input_channels_bwd, n_lstm])],
                         W_fg=[conv_W_initializer([lstm_x_fwd, lstm_y_fwd, lstm_input_channels_fwd, n_lstm]),
                               conv_W_initializer([lstm_x_bwd, lstm_y_bwd, lstm_input_channels_bwd, n_lstm])],
                         b_ci=constant([n_lstm]),
                         b_ig=constant([n_lstm]),
                         b_og=constant([n_lstm]),
                         b_fg=constant([n_lstm], 1))

        print("\tConvLSTM...")
        conv_lstm = ConvLSTMLayer(incoming=rnn_input_layer, n_units=n_lstm, **lstm_init,
                                  a_out=get_rec_attr(tf, config.lstm_act),
                                  forgetgate=getattr(config, 'forgetgate', True), store_states=config.store_states,
                                  return_states=False, precomp_fwds=False, tickerstep_biases=tf.zeros)
        layers.append(conv_lstm)

        #
        # Optional feature squashing of recurrent convLSTM connections
        #  We can use an additional convolutional layer to squash the number of LSTM output features and use its output
        #  to replace the convLSTM recurrent connections
        #
        if config.reduced_rec_lstm:
            print("\tFeatureSquashing...")
            # Define a weight kernel and create the convolutional layer for squashing
            kernel = conv_W_initializer([config.kernel_conv_out, config.kernel_conv_out,
                                         conv_lstm.get_output_shape()[-1], config.reduced_rec_lstm])
            squashed_recurrences = ConvLayer(incoming=conv_lstm, W=kernel, padding='SAME',
                                             name='ConvLayerFeatureSquashing', a=tf.nn.elu)
            layers.append(squashed_recurrences)

            print("\tConvLSTMRecurrence...")
            # Overwrite the existing ConvLSTM recurrences with the output of the feature squashing layer
            conv_lstm.add_external_recurrence(squashed_recurrences)

        #
        # Conventional ConvLayer after convLSTM as output layer (tf.identitiy output function because of cross-entropy)
        #
        print("\tConvLayer...")
        output_layer = ConvLayer(incoming=conv_lstm,
                                 W=conv_W_initializer([config.kernel_conv, config.kernel_conv,
                                                       conv_lstm.get_output_shape()[-1], n_classes]),
                                 padding='SAME', name='ConvLayerSemanticSegmentation', a=tf.identity)
        layers.append(output_layer)

        # ----------------------------------------------------------------------------------------------------------
        #  Create graph through sequence positions and ticker steps
        # ----------------------------------------------------------------------------------------------------------

        #
        # Loop through sequence positions
        #
        print("\tRNN Loop...")
        for seq_pos in range(n_seq_pos):
            with tf.name_scope("Sequence_pos_{}".format(seq_pos)):
                print("\t  seq. pos. {}...".format(seq_pos))
                # Set rnn input layer to current frame
                rnn_input_layer.update(X[:, seq_pos:seq_pos + 1, :])

                # Calculate new network state at new frame (this updates the network's hidden activations, cell states,
                # and dependencies automatically)
                output = output_layer.get_output()

        #
        # Loop through tickersteps
        #
        # Use last frame as input during ticker steps
        tickerstep_input = X[:, -1:, :]

        for tickerstep in range(config.tickersteps):
            with tf.name_scope("Tickerstep_{}".format(tickerstep)):
                print("\t  tickerstep {}...".format(tickerstep))

                # Set rnn input layer to tickerstep input
                rnn_input_layer.update(tickerstep_input)

                # Calculate new network state at new frame and activate tickerstep biases
                output = output_layer.get_output(tickerstep_nodes=True)

        print("\tDone!")

        #
        # Publish
        #
        self.X = X
        self.y_ = y_
        self.output = output
        # We will use this list of layers for regularization in the main file
        self.__layers = layers
        # We will plot some parts of the lstm, so we make it accessible as attribute
        self.lstm_layer = conv_lstm

    def get_layers(self):
        return self.__layers


class ArchitectureConvLSTMMNIST(object):
    def __init__(self, config: Config, dataset):
        """Example for convolutional network with convLSTM and convolutional output layer; Plots cell states, hidden
        states, X, y_, and a argmax over the convLSTM units outputs;

        Command-line usage:
        >>> python3 samples/main_convlstm_mnist.py --config=samples/config_convlstm_mnist.json
        """
        import TeLL

        #
        # Some convenience objects
        #
        # We will use a list to store all layers for regularization etc. (this is optional)
        layers = []

        depth = config.get_value("enc_dec_depth", 2)
        basenr_convs = config.get_value("enc_dec_conv_maps_base", 16)
        init_name = config.get_value("conv_W_initializer", "weight_xavier_conv2d")
        conv_W_initializer = getattr(TeLL.initializations, init_name)

        #
        # Create placeholders for feeding an input frame and a label at each sequence position
        #
        X_shape = dataset.mb_info['X'][0]  # dataset.X_shape is [sample, seq_pos, x, y, features)
        y_shape = dataset.mb_info['y'][0]
        frame_input_shape = X_shape[:1] + (1,) + X_shape[2:]
        n_classes = 11
        frame_output_shape = y_shape[:1] + (1,) + y_shape[2:] + (n_classes,)
        n_seq_pos = X_shape[1]
        X = tf.placeholder(tf.float32, shape=X_shape)
        y_ = tf.placeholder(tf.int32, shape=y_shape)  # dataset.y_shape is [sample, seq_pos, features)

        # ----------------------------------------------------------------------------------------------------------
        # Define network architecture
        # ----------------------------------------------------------------------------------------------------------

        #
        # Initialize input to network of shape [sample, 1, x, y, features] with zero tensor of size of a frame
        #
        rnn_input_layer = RNNInputLayer(tf.zeros(frame_input_shape, dtype=tf.float32))
        layers.append(rnn_input_layer)

        #
        # Encoder and maxpooling layers
        #
        encoders = list()
        for d in range(1, depth + 1):
            print("\tConvLayerEncoder{}...".format(d))
            layers.append(ConvLayer(incoming=layers[-1],
                                    W=conv_W_initializer([config.kernel_conv, config.kernel_conv,
                                                          layers[-1].get_output_shape()[-1], basenr_convs * (2 ** d)]),
                                    padding='SAME', name='ConvLayerEncoder{}'.format(d), a=tf.nn.elu))
            encoders.append(layers[-1])
            print("\tMaxpoolingLayer{}...".format(d))
            layers.append(MaxPoolingLayer(incoming=layers[-1], ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME',
                                          name='MaxpoolingLayer{}'.format(d)))

        #
        # ConvLSTM Layer
        #
        if config.n_lstm:
            n_lstm = config.n_lstm
            lstm_x_fwd = config.kernel_lstm_fwd
            lstm_y_fwd = config.kernel_lstm_fwd
            lstm_x_bwd = config.kernel_lstm_bwd
            lstm_y_bwd = config.kernel_lstm_bwd

            lstm_input_channels_fwd = layers[-1].get_output_shape()[-1]
            if config.reduced_rec_lstm:
                lstm_input_channels_bwd = config.reduced_rec_lstm
            else:
                lstm_input_channels_bwd = n_lstm

            lstm_init = dict(W_ci=[conv_W_initializer([lstm_x_fwd, lstm_y_fwd, lstm_input_channels_fwd, n_lstm]),
                                   conv_W_initializer([lstm_x_bwd, lstm_y_bwd, lstm_input_channels_bwd, n_lstm])],
                             W_ig=[conv_W_initializer([lstm_x_fwd, lstm_y_fwd, lstm_input_channels_fwd, n_lstm]),
                                   conv_W_initializer([lstm_x_bwd, lstm_y_bwd, lstm_input_channels_bwd, n_lstm])],
                             W_og=[conv_W_initializer([lstm_x_fwd, lstm_y_fwd, lstm_input_channels_fwd, n_lstm]),
                                   conv_W_initializer([lstm_x_bwd, lstm_y_bwd, lstm_input_channels_bwd, n_lstm])],
                             W_fg=[conv_W_initializer([lstm_x_fwd, lstm_y_fwd, lstm_input_channels_fwd, n_lstm]),
                                   conv_W_initializer([lstm_x_bwd, lstm_y_bwd, lstm_input_channels_bwd, n_lstm])],
                             b_ci=constant([n_lstm]),
                             b_ig=constant([n_lstm]),
                             b_og=constant([n_lstm]),
                             b_fg=constant([n_lstm], 1))

            print("\tConvLSTM...")
            layers.append(ConvLSTMLayer(incoming=layers[-1], n_units=n_lstm, **lstm_init,
                                        a_out=get_rec_attr(tf, config.lstm_act), forgetgate=config.forgetgate,
                                        store_states=config.store_states, tickerstep_biases=tf.zeros,
                                        output_dropout=config.lstm_output_dropout, precomp_fwds=False))
            lstm_layer = layers[-1]

            #
            # Optional feature squashing
            #
            ext_lstm_recurrence = None

            if config.reduced_rec_lstm:
                print("\tFeatureSquashing...")
                layers.append(ConvLayer(incoming=layers[-1],
                                        W=conv_W_initializer([config.kernel_conv_out, config.kernel_conv_out,
                                                              layers[-1].get_output_shape()[-1],
                                                              config.reduced_rec_lstm]),
                                        padding='SAME', name='ConvLayerFeatureSquashing', a=tf.nn.elu))
                print("\tConvLSTMRecurrence...")
                ext_lstm_recurrence = layers[-1]

            if ext_lstm_recurrence is not None:
                lstm_layer.add_external_recurrence(ext_lstm_recurrence)
        else:
            print("\tSubstituteConvLayer...")
            layers.append(ConvLayer(incoming=layers[-1],
                                    W=conv_W_initializer([config.kernel_conv, config.kernel_conv,
                                                          layers[-1].get_output_shape()[-1],
                                                          int(basenr_convs * (2 ** depth) * 4.5)]),
                                    padding='SAME', name='SubstituteConvLayer', a=tf.nn.elu))
            lstm_layer = layers[-1]

        #
        # ConvLayer for semantic segmentation
        #
        print("\tConvLayerSemanticSegmentation...")
        layers.append(ConvLayer(incoming=layers[-1],
                                W=conv_W_initializer([config.kernel_conv_out, config.kernel_conv_out,
                                                      layers[-1].get_output_shape()[-1], n_classes]),
                                padding='SAME', name='ConvLayerSemanticSegmentation', a=tf.identity))

        #
        # Upscaling layer
        #
        print("\tUpscalingLayer...")
        layers[-1] = ScalingLayer(incoming=layers[-1], size=frame_output_shape[-3:-1], name='UpscalingLayergLayer')

        output_layer = layers[-1]

        # ----------------------------------------------------------------------------------------------------------
        #  Create graph through sequence positions and ticker steps
        # ----------------------------------------------------------------------------------------------------------
        outputs_all_timesteps = []

        #
        # Loop through sequence positions
        #
        print("\tRNN Loop...")
        for seq_pos in range(n_seq_pos):
            with tf.name_scope("Sequence_pos_{}".format(seq_pos)):
                print("\t  seq. pos. {}...".format(seq_pos))
                # Set rnn input layer to current frame
                rnn_input_layer.update(X[:, seq_pos:seq_pos + 1, :])

                # Calculate new network state at new frame (this updates the network's hidden activations, cell states,
                # and dependencies automatically)
                output = output_layer.get_output()
                outputs_all_timesteps.append(output)

        #
        # Loop through tickersteps
        #
        # Use last frame as input during ticker steps
        tickerstep_input = X[:, -1:, :]

        for tickerstep in range(config.tickersteps):
            with tf.name_scope("Tickerstep_{}".format(tickerstep)):
                print("\t  tickerstep {}...".format(tickerstep))

                # Set rnn input layer to tickerstep input
                rnn_input_layer.update(tickerstep_input)

                # Calculate new network state at new frame and activate tickerstep biases
                output = output_layer.get_output(tickerstep_nodes=True)
                outputs_all_timesteps.append(output)

        print("\tDone!")

        #
        # Publish
        #
        self.X = X
        self.y_ = y_
        self.output = tf.concat(outputs_all_timesteps, axis=1, name='outputs_all_timesteps')
        pixel_weights = tf.ones_like(y_, dtype=tf.float32)
        # pixel_weights -= tf.cast(y_ == 0, dtype=tf.float32) * tf.constant(1.-0.2)
        self.pixel_weights = pixel_weights
        # We will use this list of layers for regularization in the main file
        self.__layers = layers
        # We will plot some parts of the lstm, so we make it accessible as attribute
        self.lstm_layer = lstm_layer

    def get_layers(self):
        return self.__layers


class Autoencoder(object):
    def __init__(self, config: Config, dataset):

        # We will use a list to store all layers for regularization etc. (this is optional)
        layers = []

        # Prepare xavier initialization for weights
        w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)

        #
        # Create placeholders for input data (shape: [n_samples, n_sequence_positions, n_features])
        #

        input_shapes = dataset.get_input_shapes()
        input_shapes = (input_shapes[0], None, input_shapes[2])
        X = tf.placeholder(tf.float32, shape=input_shapes['X'].shape)

        n_seq_pos = input_shapes['X'].shape[1]  # dataset.X_shape is [sample, seq_pos, features]

        # ----------------------------------------------------------------------------------------------------------
        # Define network architecture
        # ----------------------------------------------------------------------------------------------------------

        #
        # Input Layer

        input_shape = input_shapes['X'].shape
        rnn_input_layer = RNNInputLayer(tf.zeros(input_shape, dtype=tf.float32))
        layers.append(rnn_input_layer)

        #
        # LSTM Layer
        #
        print("\tLSTM... Encoder")
        # We want to modify the number of recurrent connections -> we have to specify the shape of the recurrent weights
        rec_w_shape = (config.n_lstm, config.n_lstm)
        # The forward weights can be initialized automatically, for the recurrent ones we will use our rec_w_shape
        lstm_w_enc = [w_init, w_init(rec_w_shape)]
        lstm_layer_enc = LSTMLayer(incoming=rnn_input_layer, n_units=config.n_lstm, name='LSTM',
                                   W_ci=lstm_w_enc, W_ig=lstm_w_enc, W_og=lstm_w_enc, W_fg=lstm_w_enc,
                                   b_ci=tf.zeros, b_ig=tf.zeros, b_og=tf.zeros, b_fg=tf.zeros,
                                   a_ci=tf.tanh, a_ig=tf.sigmoid, a_og=tf.sigmoid, a_fg=tf.sigmoid, a_out=tf.nn.elu,
                                   c_init=tf.zeros, h_init=tf.zeros, forgetgate=False, precomp_fwds=True,
                                   return_states=True)
        layers.append(lstm_layer_enc)

        print("\tLSTM... Decoder")
        # We want to modify the number of recurrent connections -> we have to specify the shape of the recurrent weights
        n_lstm_dec = input_shapes['X'].shape[0]
        rec_w_shape = (n_lstm_dec, n_lstm_dec)
        # The forward weights can be initialized automatically, for the recurrent ones we will use our rec_w_shape
        lstm_w_dec = [w_init, w_init(rec_w_shape)]
        lstm_layer_dec = LSTMLayer(incoming=lstm_layer_enc, n_units=n_lstm_dec, name='LSTM',
                                   W_ci=lstm_w_dec, W_ig=lstm_w_dec, W_og=lstm_w_dec, W_fg=lstm_w_dec,
                                   b_ci=tf.zeros, b_ig=tf.zeros, b_og=tf.zeros, b_fg=tf.zeros,
                                   a_ci=tf.tanh, a_ig=tf.sigmoid, a_og=tf.sigmoid, a_fg=tf.sigmoid, a_out=tf.nn.elu,
                                   c_init=tf.zeros, h_init=tf.zeros, forgetgate=False, precomp_fwds=True,
                                   return_states=True)
        layers.append(lstm_layer_dec)

        # ----------------------------------------------------------------------------------------------------------
        # Loop through sequence positions and create graph
        # ----------------------------------------------------------------------------------------------------------

        #
        # Loop through sequence positions
        #
        print("\tRNN Loop...")
        for seq_pos in range(n_seq_pos):
            with tf.name_scope("Sequence_pos_{}".format(seq_pos)):
                print("\t  seq. pos. {}...".format(seq_pos))
                # Set rnn input layer to input at current sequence position
                layers[0].update(X[:, seq_pos:seq_pos + 1])

                # Calculate new lstm state (this automatically computes all dependencies, including rec. connections)
                _ = lstm_layer_dec.get_output()

        #
        # Loop through tickersteps
        #
        # Use zeros as input during ticker steps
        tickerstep_input = tf.zeros(input_shape, dtype=tf.string,
                                    name="tickerstep_input")

        for tickerstep in range(config.tickersteps):
            with tf.name_scope("Tickerstep_{}".format(tickerstep)):
                print("\t  tickerstep {}...".format(tickerstep))
                # Set rnn input layer to input at current sequence position
                layers[0].update(tickerstep_input)

                # Calculate new lstm state (this automatically computes all dependencies, including rec. connections)
                _ = lstm_layer_dec.get_output(tickerstep_nodes=True)

        #
        # Calculate output but consider that the lstm_layer is already computed
        #
        output = lstm_layer_dec.get_output(prev_layers=[lstm_layer_dec, lstm_layer_enc])

        print("\tDone!")

        #
        # Publish
        #
        self.X = X
        self.y_ = X
        self.output = output
        self.latent_space = lstm_layer_enc.get_output(prev_layers=[lstm_layer_enc])

        # Store layers in list for regularization in main file
        self.__layers = layers

    def get_layers(self):
        return self.__layers


class AutoencoderDynamicLenght(object):
    def __init__(self, config: Config, input_shape, mb_size=1, scopename='AutoEnc', loop_parallel_iterations=1):

        self.scopename = scopename

        self.placeholders = None
        self.data_tensors = None
        self.operation_tensors = None
        self.summaries = None
        self.loop_parallel_iterations = loop_parallel_iterations

        self.lstm_network_config = config.lstm_network_config
        self.training_config = config.training_config
        self.write_histograms = config.Write_histograms

        # We will use a list to store all layers for regularization etc. (this is optional)
        layers = []

        # --------------------------------------------------------------------------------------------------------------
        # Shapes
        # --------------------------------------------------------------------------------------------------------------
        inputs_shape = [mb_size, None] + list(input_shape)
        input_shape_enc = [mb_size, 1] + list(input_shape[:-1]) + [1]
        output_shape_enc = [mb_size, 1] + list(input_shape[:-1]) + [self.lstm_network_config['n_latent_output_units']]

        #
        # Create placeholders for input data (shape: [n_samples, n_sequence_positions, n_features])
        #

        input_placeholder = tf.placeholder(tf.float32, shape=inputs_shape)
        sequence_length_placeholder = tf.placeholder(shape=(), dtype=tf.int32, name='game_length_placeholder')

        # --------------------------------------------------------------------------------------------------------------
        # Input to LSTM network
        # --------------------------------------------------------------------------------------------------------------
        # Create input layers (these will be set dynamically in LSTM loop)
        enc_lstm_input_layer = RNNInputLayer(tf.zeros(input_shape_enc, dtype=tf.float32))
        dec_lstm_input_layer = RNNInputLayer(tf.zeros(output_shape_enc, dtype=tf.float32))

        with tf.variable_scope(self.scopename, reuse=tf.AUTO_REUSE):

            # ----------------------------------------------------------------------------------------------------------
            # LSTM network
            # ----------------------------------------------------------------------------------------------------------
            print("Building Autoencoder LSTM network...")

            #
            # Initializations
            #
            n_latent_output_units = self.lstm_network_config['n_latent_output_units']
            layerspecs = self.lstm_network_config['layers']
            init_specs = self.lstm_network_config['initializations_encoder']
            init_specs_dec = self.lstm_network_config['initializations_decoder']

            lstm_layer_index = [i for i, l in enumerate(layerspecs) if l['type'] == 'LSTMLayer'][0]
            lstm_specs = layerspecs[lstm_layer_index]
            n_lstm = n_latent_output_units

            lstm_w_init = lambda scale: lambda *args, **kwargs: tf.truncated_normal(*args, **kwargs) * scale
            truncated_normal_init = lambda mean, stddev: \
                lambda *args, **kwargs: tf.truncated_normal(mean=mean, stddev=stddev, *args, **kwargs)

            #
            # Initialization/activation functions
            #
            if lstm_specs['a_out'] == 'linear':
                lstm_a_out = tf.identity
            elif lstm_specs['a_out'] == 'tanh':
                lstm_a_out = tf.tanh

            og_bias = truncated_normal_init(mean=init_specs['og_bias'], stddev=0.1)
            ig_bias = truncated_normal_init(mean=init_specs['ig_bias'], stddev=0.1)
            ci_bias = truncated_normal_init(mean=init_specs['ci_bias'], stddev=0.1)
            fg_bias = truncated_normal_init(mean=init_specs['fg_bias'], stddev=0.1)

            #
            # Layers setup
            #
            print("\tLSTM network for RR...")
            with tf.variable_scope('lstmnet', reuse=tf.AUTO_REUSE):

                # Store all layers in a list
                enc_layers = [enc_lstm_input_layer]

                #
                # Create layers before LSTM
                #
                # enc_layers += layers_from_specs(incoming=enc_lstm_input_layer, layerspecs=layerspecs[:lstm_layer_index])
                enc_lstm_input_layer = enc_layers[-1]

                #
                # Create LSTM decoder layer
                #

                # Weight initialization for encoder LSTM
                # Reducing the connectivity of the LSTM might help for some tasks
                no_fwd_connection = tf.get_variable(initializer=constant([enc_lstm_input_layer.get_output_shape()[-1],
                                                                          n_lstm], 0),
                                                    trainable=False, name='no_fwd_connection')
                no_bwd_connection = tf.get_variable(initializer=constant([n_lstm, n_lstm], 0), trainable=False,
                                                    name='no_bwd_connection')

                # If w init variance is -1, disable the connections and if using sigma
                w_og = [lstm_w_init(w) if w != -1 else z for w, z in zip(init_specs['w_og'],
                                                                         [no_fwd_connection, no_bwd_connection])]
                w_fg = [lstm_w_init(w) if w != -1 else z for w, z in zip(init_specs['w_fg'],
                                                                         [no_fwd_connection, no_bwd_connection])]
                w_ig = [lstm_w_init(w) if w != -1 else z for w, z in zip(init_specs['w_ig'],
                                                                         [no_fwd_connection, no_bwd_connection])]
                w_ci = [lstm_w_init(w) if w != -1 else z for w, z in zip(init_specs['w_ci'],
                                                                         [no_fwd_connection, no_bwd_connection])]
                a_ci = tf.tanh
                b_ci = ci_bias([n_lstm])

                forgetgate_active = not all([w == -1 for w in init_specs['w_fg']])
                outputgate_active = not all([w == -1 for w in init_specs['w_og']])

                if forgetgate_active:
                    b_fg = fg_bias([n_lstm])
                    a_fg = tf.sigmoid
                else:
                    b_fg = tf.get_variable(initializer=constant([n_lstm], 1), trainable=False, name='no_b_fg')
                    a_fg = tf.identity

                if outputgate_active:
                    b_og = og_bias([n_lstm])
                    a_og = tf.sigmoid
                else:
                    b_og = tf.get_variable(initializer=constant([n_lstm], 1), trainable=False, name='no_b_og')
                    a_og = tf.identity

                ### LSTM Encoder

                enc_lstm_layer = LSTMLayer(incoming=enc_lstm_input_layer, n_units=n_lstm,
                                           name='LSTM_Encoder',
                                           W_ci=w_ci, W_ig=w_ig, W_og=w_og, W_fg=w_fg,
                                           b_ci=b_ci, b_ig=ig_bias([n_lstm]),
                                           b_og=b_og, b_fg=b_fg,
                                           a_ci=a_ci, a_ig=tf.sigmoid, a_og=a_og,
                                           a_fg=a_fg, a_out=lstm_a_out,
                                           c_init=tf.zeros, h_init=tf.zeros, forgetgate=forgetgate_active,
                                           precomp_fwds=False, store_states=True, return_states=False)

                enc_layers.append(enc_lstm_layer)
                #
                # Create layers after LSTM
                #
                if lstm_layer_index + 1 < len(layerspecs):
                    enc_layers += layers_from_specs(incoming=enc_layers[-1],
                                                    layerspecs=layerspecs[lstm_layer_index + 1:])

                ########### End Encoder ################

                latent_space_layer = enc_layers[-1]

                ### LSTM Decoder

                # Weight initialization for encoder LSTM
                # Reducing the connectivity of the LSTM might help for some tasks
                no_fwd_connection_d = tf.get_variable(initializer=constant([latent_space_layer.get_output_shape()[-1],
                                                                            n_lstm], 0),
                                                      trainable=False, name='no_fwd_connection_dec')
                no_bwd_connection_d = tf.get_variable(initializer=constant([n_lstm, n_lstm], 0), trainable=False,
                                                      name='no_bwd_connection_dec')

                # If w init variance is -1, disable the connections and if using sigma
                w_og_d = [lstm_w_init(w) if w != -1 else z for w, z in zip(init_specs_dec['w_og'],
                                                                           [no_fwd_connection_d, no_bwd_connection_d])]
                w_fg_d = [lstm_w_init(w) if w != -1 else z for w, z in zip(init_specs_dec['w_fg'],
                                                                           [no_fwd_connection_d, no_bwd_connection_d])]
                w_ig_d = [lstm_w_init(w) if w != -1 else z for w, z in zip(init_specs_dec['w_ig'],
                                                                           [no_fwd_connection_d, no_bwd_connection_d])]
                w_ci_d = [lstm_w_init(w) if w != -1 else z for w, z in zip(init_specs_dec['w_ci'],
                                                                           [no_fwd_connection_d, no_bwd_connection_d])]
                # This part can remain the same
                a_ci = tf.tanh
                b_ci = ci_bias([n_lstm])

                forgetgate_active = not all([w == -1 for w in init_specs_dec['w_fg']])
                outputgate_active = not all([w == -1 for w in init_specs_dec['w_og']])

                if forgetgate_active:
                    b_fg = fg_bias([n_lstm])
                    a_fg = tf.sigmoid
                else:
                    b_fg = tf.get_variable(initializer=constant([n_lstm], 1), trainable=False, name='no_b_fg')
                    a_fg = tf.identity

                if outputgate_active:
                    b_og = og_bias([n_lstm])
                    a_og = tf.sigmoid
                else:
                    b_og = tf.get_variable(initializer=constant([n_lstm], 1), trainable=False, name='no_b_og')
                    a_og = tf.identity

                dec_layers = []

                dec_lstm_layer = LSTMLayer(incoming=dec_lstm_input_layer, n_units=n_lstm,
                                           name='LSTM_Decoder',
                                           W_ci=w_ci_d, W_ig=w_ig_d, W_og=w_og_d, W_fg=w_fg_d,
                                           b_ci=b_ci, b_ig=ig_bias([n_lstm]),
                                           b_og=b_og, b_fg=b_fg,
                                           a_ci=a_ci, a_ig=tf.sigmoid, a_og=a_og,
                                           a_fg=a_fg, a_out=lstm_a_out,
                                           c_init=tf.zeros, h_init=tf.zeros, forgetgate=forgetgate_active,
                                           precomp_fwds=False, store_states=True, return_states=False)

                dec_layers.append(dec_lstm_layer)
                #
                # Create layers after LSTM
                #
                if lstm_layer_index + 1 < len(layerspecs):
                    dec_layers += layers_from_specs(incoming=dec_layers[-1],
                                                    layerspecs=layerspecs[lstm_layer_index + 1:])

                #
                # Create Decoder output layer
                #
                dec_layers.append(DenseLayer(incoming=dec_layers[-1], n_units=inputs_shape[-1], a=tf.identity,
                                             W=lstm_w_init(1), b=constant([inputs_shape[-1]], 0.),
                                             name='DenseCAout'))
                output_layer = dec_layers[-1]

                ########### End Decoder ################

        # --------------------------------------------------------------------------------------------------------------
        #  LSTM network loop
        # --------------------------------------------------------------------------------------------------------------
        # with tf.device("/cpu:0"):
        #
        # Layers that require sequential computation (i.e. after LSTM incl. LSTM) will be computed in tf.while loop
        #

        print("\tSetting up LSTM loop...")
        g = tf.get_default_graph()

        #
        # Get layers that require sequential computation (i.e. after LSTM incl. LSTM but excluding output layer)
        #
        enc_lstm_layer_position = [i for i, l in enumerate(enc_layers)
                                   if isinstance(l, LSTMLayer) or isinstance(l, LSTMLayerGetNetInput)][0]
        dec_lstm_layer_position = [i for i, l in enumerate(dec_layers)
                                   if isinstance(l, LSTMLayer) or isinstance(l, LSTMLayerGetNetInput)][0]

        enc_layers_head = enc_layers[enc_lstm_layer_position + 1:-1]
        dec_layers_head = dec_layers[dec_lstm_layer_position + 1:-1]

        n_timesteps = sequence_length_placeholder - 1

        with tf.name_scope("EncoderLoopLSTM"):
            #
            # Ending condition
            #
            def cond_encoder(time, *args):
                """Break if game is over by looking at n_timesteps"""
                return ~tf.greater(time, n_timesteps)

            #
            # Loop body for encoder
            #
            # Create initial tensors
            init_tensors = OrderedDict([
                ('time', tf.constant(0, dtype=tf.int32)),
                ('enc_lstm_internals', tf.expand_dims(tf.stack([enc_lstm_layer.c[-1], enc_lstm_layer.c[-1],
                                                                enc_lstm_layer.c[-1], enc_lstm_layer.c[-1],
                                                                enc_lstm_layer.c[-1], enc_lstm_layer.c[-1],
                                                                enc_lstm_layer.c[-1]], axis=-1), axis=1)),
                # We add to the internals state and action
                ('enc_lstm_h', tf.expand_dims(enc_lstm_layer.h[-1], axis=1)),
                ('latent_space', tf.zeros([s if s >= 0 else 1 for s in latent_space_layer.get_output_shape()])),
                ('time_constant', tf.constant(1, dtype=tf.int32))
            ])
            if len(enc_layers_head) > 0:
                init_tensors.update(OrderedDict(
                    [('enc_dense_layer_{}'.format(i),
                      tf.zeros([s for s in l.get_output_shape() if s >= 0], dtype=tf.float32))
                     for i, l in enumerate(enc_layers_head)]))

            # Get initial tensor shapes in tf format
            init_shapes = OrderedDict([
                ('time', init_tensors['time'].get_shape()),
                ('enc_lstm_internals', tensor_shape_with_flexible_dim(init_tensors['enc_lstm_internals'], dim=1)),
                ('enc_lstm_h', tensor_shape_with_flexible_dim(init_tensors['enc_lstm_h'], dim=1)),
                ('latent_space', tensor_shape_with_flexible_dim(init_tensors['latent_space'], dim=1)),
                ('time_constant', init_tensors['time_constant'].get_shape())

            ])

            if len(enc_layers_head) > 0:
                init_shapes.update(OrderedDict(
                    [('enc_dense_layer_{}'.format(i), init_tensors['enc_dense_layer_{}'.format(i)].get_shape())
                     for i, l in enumerate(enc_layers_head)]))

            def body_encoder(time, enc_lstm_internals, enc_lstm_h, latent_space, time_constant, *args):
                """Loop over frames and additional inputs, compute network outputs and store hidden states and
                activations for debugging/plotting and integrated gradients calculation
                Loop for the encoder """
                time_index = time
                #
                # Set state and state-deltas as network input
                #
                enc_lstm_input_layer.update(tf.cast(tf.expand_dims(input_placeholder[:, time_index, ..., -1:],
                                                                   axis=1), dtype=tf.float32))
                #
                # Update LSTM cell-state and output with states from last timestep, encoder
                #
                enc_lstm_layer.c[-1], enc_lstm_layer.h[-1] = enc_lstm_internals[:, -1, :, -1], enc_lstm_h[:, -1, :]
                #
                # Calculate latent space ( we will use the last timestep )
                #
                latent_space = tf.concat([latent_space,
                                          latent_space_layer.get_output()], axis=1)  # prev_layers ?
                #
                # Store LSTM states for all timesteps for visualization
                #
                enc_lstm_internals = tf.concat([enc_lstm_internals,
                                                tf.expand_dims(
                                                    tf.stack([enc_lstm_layer.ig[-1], enc_lstm_layer.og[-1],
                                                              enc_lstm_layer.ci[-1], enc_lstm_layer.fg[-1],
                                                              enc_lstm_layer.fg[-1], enc_lstm_layer.fg[-1],
                                                              enc_lstm_layer.c[-1]], axis=-1),
                                                    axis=1)],
                                               axis=1)
                enc_lstm_h = tf.concat([enc_lstm_h, tf.expand_dims(enc_lstm_layer.h[-1], axis=1)], axis=1)
                #
                # Store output of optional layers above LSTM for debugging
                #
                enc_layers_head_activations = [l.out for l in enc_layers_head]
                #
                # Increment time
                #
                time += time_constant

                return [time, enc_lstm_internals, enc_lstm_h, latent_space, time_constant,
                        *enc_layers_head_activations]

            wl_ret = tf.while_loop(cond=cond_encoder, body=body_encoder, loop_vars=tuple(init_tensors.values()),
                                   shape_invariants=tuple(init_shapes.values()),
                                   parallel_iterations=self.loop_parallel_iterations, back_prop=True, swap_memory=True)

            # Re-Associate returned tensors with keys
            encoder_returns = OrderedDict(zip(init_tensors.keys(), wl_ret))

            # Remove initialization timestep
            encoder_returns['enc_lstm_internals'] = encoder_returns['enc_lstm_internals'][:, 1:]
            encoder_returns['enc_lstm_h'] = encoder_returns['enc_lstm_h'][:, 1:]
            encoder_returns['latent_space'] = encoder_returns['latent_space'][:, 1:]

            if len(enc_layers_head) > 0:
                for i, l in enumerate(enc_layers_head):
                    encoder_returns['enc_dense_layer_{}'.format(i)] = encoder_returns['enc_dense_layer_{}'.format(i)][:,
                                                                      1:]

        with tf.name_scope("DecoderLoopLSTM"):

            #
            # Ending condition
            #
            def cond_decoder(time, *args):
                """Break if game is over by looking at n_timesteps"""
                return ~tf.greater(time, n_timesteps)

            #
            # Loop body for decoder
            #
            # Create initial tensors
            # latent_space_layer
            # Here we initialize the cell states with the latent space
            init_tensors_dec = OrderedDict([
                ('time', tf.constant(0, dtype=tf.int32)),
                ('dec_lstm_internals', tf.expand_dims(tf.stack([dec_lstm_layer.c[-1], dec_lstm_layer.c[-1],
                                                                dec_lstm_layer.c[-1], dec_lstm_layer.c[-1],
                                                                dec_lstm_layer.c[-1], dec_lstm_layer.c[-1],
                                                                encoder_returns['latent_space'][:, -1, :]], axis=-1),
                                                      axis=1)),
                # We add to the internals state and action
                ('dec_lstm_h', tf.expand_dims(dec_lstm_layer.h[-1], axis=1)),
                ('output', tf.zeros([s if s >= 0 else 1 for s in output_layer.get_output_shape()])),
                ('time_constant', tf.constant(1, dtype=tf.int32))
            ])

            # Get initial tensor shapes in tf format
            init_shapes_dec = OrderedDict([
                ('time', init_tensors_dec['time'].get_shape()),
                ('dec_lstm_internals',
                 tensor_shape_with_flexible_dim(init_tensors_dec['dec_lstm_internals'], dim=1)),
                ('dec_lstm_h', tensor_shape_with_flexible_dim(init_tensors_dec['dec_lstm_h'], dim=1)),
                ('output', tensor_shape_with_flexible_dim(init_tensors_dec['output'], dim=1)),
                ('time_constant', init_tensors['time_constant'].get_shape())

            ])

            def body_decoder(time, dec_lstm_internals, dec_lstm_h, output, time_constant, *args):
                """Loop over frames and additional inputs, compute network outputs and store hidden states and
                activations for debugging/plotting and integrated gradients calculation
                Loop for the encoder """
                time_index = time

                #
                # Input is zero. So no need to update. Latent space is fed directly to the cell memory as initialization
                #
                dec_lstm_input_layer.update(
                    tf.cast(tf.expand_dims(tf.zeros_like(encoder_returns['latent_space'][:, 0, :]),
                                           axis=1), dtype=tf.float32))
                #
                # Update LSTM cell-state and output with states from last timestep, encoder
                #
                dec_lstm_layer.c[-1], dec_lstm_layer.h[-1] = dec_lstm_internals[:, -1, :, -1], dec_lstm_h[:, -1, :]
                #
                # Calculate latent space ( we will use the last timestep )
                #
                output = tf.concat([output, output_layer.get_output(prev_layers=[enc_layers])], axis=1)  # prev_layers ?
                #
                # Store LSTM states for all timesteps for visualization
                #
                dec_lstm_internals = tf.concat([dec_lstm_internals,
                                                tf.expand_dims(
                                                    tf.stack([dec_lstm_layer.ig[-1], dec_lstm_layer.og[-1],
                                                              dec_lstm_layer.ci[-1], dec_lstm_layer.fg[-1],
                                                              dec_lstm_layer.fg[-1], dec_lstm_layer.fg[-1],
                                                              dec_lstm_layer.c[-1]], axis=-1),
                                                    axis=1)],
                                               axis=1)
                dec_lstm_h = tf.concat([dec_lstm_h, tf.expand_dims(dec_lstm_layer.h[-1], axis=1)], axis=1)

                #
                # Increment time
                #
                time += time_constant

                return [time, dec_lstm_internals, dec_lstm_h, output, time_constant]

            with tf.control_dependencies([encoder_returns['latent_space']]):
                wl_ret_dec = tf.while_loop(cond=cond_decoder, body=body_decoder,
                                           loop_vars=tuple(init_tensors_dec.values()),
                                           shape_invariants=tuple(init_shapes_dec.values()),
                                           parallel_iterations=self.loop_parallel_iterations, back_prop=True,
                                           swap_memory=True)

            # Re-Associate returned tensors with keys
            decoder_returns = OrderedDict(zip(init_tensors_dec.keys(), wl_ret_dec))

            # Remove initialization timestep
            decoder_returns['dec_lstm_internals'] = decoder_returns['dec_lstm_internals'][:, 1:]
            decoder_returns['dec_lstm_h'] = decoder_returns['dec_lstm_h'][:, 1:]
            decoder_returns['output'] = decoder_returns['output'][:, 1:]

        # --------------------------------------------------------------------------------------------------------------
        #  Loss and Update Steps for Autoencoder Network
        # --------------------------------------------------------------------------------------------------------------
        print("\tSetting up RR updates...")

        #
        # Error for reconstruction (main target)
        #
        loss_last_timesteps = tf.reduce_sum(tf.square(decoder_returns['output'][:, -1] - input_placeholder[:, -1]))

        #
        # Add regularization penalty
        #
        layers_to_train = enc_layers + dec_layers
        ae_reg_penalty = regularize(layers=layers_to_train, l1=self.training_config['l1'],
                                    l2=self.training_config['l2'], regularize_weights=True, regularize_biases=True)

        #
        # RR Update
        #
        ae_loss = (loss_last_timesteps + ae_reg_penalty)

        enc_trainables_lstm = [t for t in tf.trainable_variables() if t.name.find('LSTM_Encoder') != -1]
        dec_trainables_lstm = [t for t in tf.trainable_variables() if t.name.find('LSTM_Decoder') != -1]

        ae_trainables = enc_trainables_lstm + dec_trainables_lstm

        # Get gradients
        ae_grads = tf.gradients(tf.reduce_mean(ae_loss), ae_trainables)

        if self.training_config['clip_gradients']:
            ae_grads, _ = tf.clip_by_global_norm(ae_grads, self.training_config['clip_gradients'])

        # Set up optimizer
        ae_update = tf.constant(0)
        if self.training_config['optimizer_params']['learning_rate'] != 0:
            with tf.variable_scope('ae_update', reuse=tf.AUTO_REUSE):
                optimizer = getattr(tf.train,
                                    self.training_config['optimizer'])(**self.training_config['optimizer_params'])
                ae_update = optimizer.apply_gradients(zip(ae_grads, ae_trainables))

        # --------------------------------------------------------------------------------------------------------------
        #  TF-summaries
        # --------------------------------------------------------------------------------------------------------------
        ae_summaries = []

        ae_summaries.append(tf.summary.scalar("AE/return_pred_loss_last_timestep", loss_last_timesteps))

        ae_summaries.append(tf.summary.scalar("AE/ae_reg_penalty", ae_reg_penalty))

        if self.write_histograms:
            ae_summaries += [tf.summary.histogram("activations/AE/{}".format(n),
                                                  values=encoder_returns['enc_lstm_internals'][0, -1, 0, i])
                             for i, n in enumerate(['enc_lstm_ig', 'enc_lstm_og', 'enc_lstm_ci', 'enc_lstm_fg'])]

            ae_summaries.append(tf.summary.histogram("activations/AE/lstm_h", encoder_returns['enc_lstm_h'][0, -1, :]))

            ae_summaries += [tf.summary.histogram("activations/AE/{}".format(n),
                                                  values=decoder_returns['dec_lstm_internals'][0, -1, 0, i])
                             for i, n in enumerate(['dec_lstm_ig', 'dec_lstm_og', 'dec_lstm_ci', 'dec_lstm_fg'])]

            ae_summaries.append(tf.summary.histogram("activations/AE/lstm_h", decoder_returns['dec_lstm_h'][0, -1, :]))

            ae_summaries += [tf.summary.histogram("gradients/AE/{}".format(t.name), values=g)
                             for g, t in zip(ae_grads, ae_trainables)]
            ae_summaries += [tf.summary.histogram("weights/AE/{}".format(t.name), values=t) for t in ae_trainables]
            ae_summaries += [tf.summary.histogram("weights/AE/{}".format(t.name), values=t) for t in ae_trainables]

        # --------------------------------------------------------------------------------------------------------------
        #  Publish
        # --------------------------------------------------------------------------------------------------------------

        # Placeholders
        placeholders = OrderedDict(
            input_placeholder=input_placeholder,
            sequence_length_placeholder=sequence_length_placeholder
        )

        # Data
        data_tensors = OrderedDict(
            lstm_internals_enc=encoder_returns['enc_lstm_internals'][0],
            lstm_internals_dec=decoder_returns['dec_lstm_internals'][0],
            lstm_h_enc=encoder_returns['enc_lstm_h'],
            lstm_h_dec=decoder_returns['dec_lstm_h'],
            latent_space=encoder_returns['latent_space'],
            loss=ae_loss,
            loss_last_time_prediction=loss_last_timesteps,
            reg_loss=ae_reg_penalty
        )

        # Operations
        operation_tensors = OrderedDict(
            ae_update=ae_update
        )

        # Summaries
        summaries = OrderedDict(
            all_summaries=tf.summary.merge_all(),
            ae_summaries=tf.summary.merge(ae_summaries)
        )

        self.placeholders = placeholders
        self.data_tensors = data_tensors
        self.operation_tensors = operation_tensors
        self.summaries = summaries

    def get_layers(self):
        return self.__layers
