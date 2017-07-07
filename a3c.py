# ----------------------------------------------------
# RL-planning (VIN + A3C)
# Author: xiangyu.liu@horizon-robotics
# Date:
# Filename: a3c.py
# ----------------------------------------------------

"""Defines policy networks for asynchronous advantage actor-critic architectures.
Heavily influenced by DeepMind's seminal paper 'Asynchronous Methods for Deep Reinforcement
Learning' (Mnih et al., 2016).
"""

import numpy as np
import tensorflow as tf

from constants import CONFIG, ACTION_SIZE, FRAME
from constants import MAP_SIZE, INPUT_DICT, SCREEN_CAPTURE, COLLISION_QUIT, STATIC, SUB_SEQ_DIM


def _flipkernel(kern):
    return kern[(slice(None, None, -1),) * 2 + (slice(None), slice(None))]


def _conv2d_flipkernel(x, k, stride=1, activation_fn=tf.identity, name=None):
    return activation_fn(tf.nn.conv2d(x, _flipkernel(k), name=name,
                                      strides=(1, stride, stride, 1), padding='SAME'))


def _fc_variable(weight_shape, name=None):
    input_channels, output_channels = weight_shape[0], weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]
    # weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d), name=name+"_w")
    # bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d), name=name+"_b")
    weight = tf.Variable(tf.truncated_normal(weight_shape, stddev=0.01), dtype=tf.float32, name=name + "_w")
    # bias = tf.Variable(tf.truncated_normal(bias_shape), dtype=tf.float32, name=name + "_b")
    bias = tf.Variable(tf.zeros(bias_shape), dtype=tf.float32, name=name + "_b")
    return weight, bias


def _conv_variable(weight_shape, name=None):
    w, h = weight_shape[0], weight_shape[1]
    input_channels, output_channels = weight_shape[2], weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]
    # weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d), name=name+"_w")
    # bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d), name=name+"_b")
    weight = tf.Variable(tf.truncated_normal(weight_shape, stddev=0.01), dtype=tf.float32, name=name+"_w")
    # bias = tf.Variable(tf.truncated_normal(bias_shape), dtype=tf.float32, name=name+"_b")
    bias = tf.Variable(tf.zeros(bias_shape), dtype=tf.float32, name=name + "_b")
    return weight, bias


class PolicyNetwork():
    def __init__(self, USE_LSTM=False):
        """Defines a policy network implemented as a VIN convolutional neural network.
        """

        k = CONFIG['k']  # Number of value iterations performed
        ch_i = CONFIG['ch_i']  # Channels in input layer
        ch_q = CONFIG['ch_q']  # Channels in q layer (~actions)
        kern = CONFIG['kern']  # Kernel size of conv
        num_action = ACTION_SIZE

        self.s = tf.placeholder(tf.float32, [None, MAP_SIZE, MAP_SIZE, ch_i], name='Input_States')
        self.pos = tf.placeholder(tf.int32, shape=[None, 2], name="pos")
        ##
        self.subsequence = tf.placeholder(tf.float32, [None, None, MAP_SIZE, MAP_SIZE, ch_i], name='sub-sequence')
        ##
        # weights from inputs to q layer (~reward in Bellman equation)
        w0, bias0 = _conv_variable([kern, kern, ch_i, 250], name="w0")
        # w1, bias1 = _conv_variable([kern, kern, 64, 16], name="w1")
        w2, bias2 = _conv_variable([1, 1, 250, 1], name="w_r")
        w_q, bias3 = _conv_variable([kern, kern, 1, ch_q], name="w_q")
        # feedback weights from v layer into q layer (~transition probabilities in
        # Bellman equation)
        w_fb, bias4 = _conv_variable([kern, kern, 1, ch_q], name="w_fb")
        w_o, bias5 = _fc_variable([ch_q, num_action], name="w_o")
        w_v, bias6 = _fc_variable([ch_q, 1], name="w_v")

        # initial conv layer over image+reward prior
        h0 = _conv2d_flipkernel(self.s, w0, name="h0") + bias0
        # h1 = _conv2d_flipkernel(h0, w1, name="h1")

        r = _conv2d_flipkernel(h0, w2, name="r")
        q = _conv2d_flipkernel(r, w_q, name="q")
        v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

        # iteration = tf.where(tf.less(k - 1, tf.shape(self.subsequence)[1]),
        #                      k - 1, tf.shape(self.subsequence)[1])

        for i in range(0, k - 1):
            rv = tf.concat([r, v], 3)
            wwfb = tf.concat([w_q, w_fb], 2)
            q = _conv2d_flipkernel(rv, wwfb, name="q")
            v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")
            if i < SUB_SEQ_DIM:
                s_tmp = self.subsequence[:, i, :, :, :]
                # assert tf.shape(self.s) == tf.shape(s_tmp
                h0 = _conv2d_flipkernel(s_tmp, w0, name="h0") + bias0
                r = _conv2d_flipkernel(h0, w2, name="r")
        # do one last convolution (NHWC)
        q = _conv2d_flipkernel(tf.concat([r, v], 3),
                               tf.concat([w_q, w_fb], 2), name="q")

        bs = tf.shape(q)[0]
        rprn = tf.reshape(tf.range(bs), [-1, 1])
        idx_in = tf.concat([rprn, self.pos], axis=1)
        # assert idx_in.get_shape().as_list() == [bs, 3]
        q_out = tf.reshape(tf.gather_nd(q, idx_in, name="q_out"), [-1, ch_q])

        # add_to_collection
        tf.add_to_collection('r', r)
        tf.add_to_collection('v', v)

        # policy (output)
        self.action_logits = tf.matmul(q_out, w_o) + bias5
        self.action = tf.squeeze(tf.multinomial(
            self.action_logits - tf.reduce_max(self.action_logits, 1, keep_dims=True), 1))
        # value (output)
        v_ = tf.matmul(q_out, w_v) + bias6
        self.value = tf.squeeze(v_)
        self.parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            tf.get_variable_scope().name)

    def sample_action(self, state, pos, subsequence):
        """Samples an action for the specified state from the learned mixed strategy.
        Args:
            state: State of the environment.
            pos: agent's position
            subsequence: subsequent states (from simulator, ground-truth)
        Returns:
            An action, the value of the specified state
        """

        sess = tf.get_default_session()
        feed_dict = {self.s: [state], self.pos: [pos], self.subsequence: [subsequence]}
        return sess.run((self.action, self.value), feed_dict)

    def estimate_value(self, state, pos, subsequence):
        """Estimates the value of the specified state.
        Args:
            state: State of the environment.
            pos: agent's position
            subsequence: subsequent states (from simulator, ground-truth)
        Returns:
            The value of the specified state.
        """

        sess = tf.get_default_session()
        feed_dict = {self.s: [state], self.pos: [pos], self.subsequence: [subsequence]}
        return sess.run(self.value, feed_dict)
