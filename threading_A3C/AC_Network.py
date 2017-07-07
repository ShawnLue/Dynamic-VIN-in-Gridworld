import tensorflow as tf
import numpy as np


from constants import FRAME
from constants import MAP_SIZE

from constants import CONFIG

# Actor-Critic Network Base Class
# (Policy network and Value network)


class GameACNetwork(object):

    def __init__(self, action_size, thread_index,  # -1 for global
                 device="/cpu_0"):
        self._action_size = action_size
        self._thread_index = thread_index
        self._device = device

    # modification?
    def prepare_loss(self, entropy_beta):
        with tf.device(self._device):
            # take action (input for policy)
            self.a = tf.placeholder(tf.float32, name="action", shape=[None, self._action_size])
            # temporary difference (R - V) (input for policy)
            self.td = tf.placeholder(tf.float32, name="td", shape=[None])

            # avoid NaN with clipping when value in pi becomes zero
            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))
            # policy entropy
            entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)

            # policy loss (output)  (Adding minus, because the original paper's objective function
            # is for gradient ascent, but we use gradient descent optimizer.)
            policy_loss = -tf.reduce_sum(tf.reduce_sum(tf.mul(log_pi, self.a),
                                                       reduction_indices=1) * self.td + entropy * entropy_beta)
            # R (input for value)
            self.r = tf.placeholder(tf.float32, name="r", shape=[None])

            # value loss (output)
            # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
            value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

            # gradient of policy and value are summed up
            self.total_loss = policy_loss + value_loss

    def run_policy_and_value(self, sess, s_t, pos):
        raise NotImplementedError()

    def run_policy(self, sess, s_t, pos):
        raise NotImplementedError()

    def run_value(self, sess, s_t, pos):
        raise NotImplementedError()

    def get_vars(self):
        raise NotImplementedError()

    def sync_from(self, src_network, name=None):
        src_vars = src_network.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []
        with tf.device(self._device):
            with tf.name_scope(name, "GameACNetwork", []) as name:
                for (src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)
                return tf.group(*sync_ops, name=name)

    def _fc_variable(self, weight_shape, name=None):
        input_channels, output_channels = weight_shape[0], weight_shape[1]
        d = 1.0 / np.sqrt(input_channels)
        bias_shape = [output_channels]
        weight = tf.Variable(0.01 * tf.random_normal(weight_shape), dtype=tf.float32, name=name + "_w")
        # weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d), dtype=tf.float32, name=name+"_w")
        bias = tf.Variable(0.01 * tf.random_normal(bias_shape), dtype=tf.float32, name=name + "_b")
        # bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d), dtype=tf.float32, name=name+"_b")
        return weight, bias

    def _conv_variable(self, weight_shape, name=None):
        w, h = weight_shape[0], weight_shape[1]
        input_channels, output_channels = weight_shape[2], weight_shape[3]
        d = 1.0 / np.sqrt(input_channels * w * h)
        bias_shape = [output_channels]
        weight = tf.Variable(0.01 * tf.random_normal(weight_shape), dtype=tf.float32, name=name + "_w")
        # weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d), dtype=tf.float32, name=name+"_w")
        bias = tf.Variable(0.01 * tf.random_normal(bias_shape), dtype=tf.float32, name=name + "_b")
        # bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d), dtype=tf.float32, name=name+"_b")
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def _flipkernel(self, kern):
        return kern[(slice(None, None, -1),) * 2 + (slice(None), slice(None))]

    def _conv2d_flipkernel(self, x, k, name=None):
        return tf.nn.conv2d(x, self._flipkernel(k), name=name, strides=(1, 1, 1, 1), padding='SAME')


# Actor-Critic FF Network
class GameAC_FF_Network(GameACNetwork):

    def __init__(self, action_size, thread_index, device="/cpu:0"):
        GameACNetwork.__init__(self, action_size, thread_index, device)

        scope_name = "net_" + str(self._thread_index)
        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            self.W_conv1, self.b_conv1 = self._conv_variable(
                [1, 1, FRAME + 1, 64])  # stride=1
            self.W_conv2, self.b_conv2 = self._conv_variable(
                [1, 1, 64, 16])  # stride=1
            self.W_conv3, self.b_conv3 = self._conv_variable(
                [1, 1, 16, 1])  # stride=1

            self.W_fc1, self.b_fc1 = self._fc_variable([2500, 256])  # reset

            # weight for policy output layer
            self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size])

            # weight for value output layer
            self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])

            # state (input)
            self.s = tf.placeholder(tf.float32, name="state", shape=[None, MAP_SIZE, MAP_SIZE, FRAME + 1])
            h_conv1 = tf.nn.relu(self._conv2d(self.s, self.W_conv1, 1) + self.b_conv1)
            h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 1) + self.b_conv2)
            h_conv3 = tf.nn.relu(self._conv2d(h_conv2, self.W_conv3, 1) + self.b_conv3)

            h_conv3_flat = tf.reshape(h_conv3, [-1, 2500])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.W_fc1) + self.b_fc1)

            # policy (output)
            self.pi = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
            # value (output)
            v_ = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
            self.v = tf.reshape(v_, [-1])

    def run_policy_and_value(self, sess, s_t, pos=None):
        pi_out, v_out = sess.run([self.pi, self.v], feed_dict={self.s: [s_t]})
        return pi_out[0], v_out[0]

    def run_policy(self, sess, s_t, pos=None):
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]

    def run_value(self, sess, s_t, pos=None):
        v_out = sess.run(self.v, feed_dict={self.s: [s_t]})
        return v_out[0]

    def get_vars(self):
        return [self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_conv3, self.b_conv3,
                self.W_fc1, self.b_fc1,
                self.W_fc2, self.b_fc2,
                self.W_fc3, self.b_fc3]


# Actor-Critic LSTM Network
class GameAC_LSTM_Network(GameACNetwork):

    def __init__(self, action_size, thread_index, device="/cpu:0"):
        GameACNetwork.__init__(self, action_size, thread_index, device)

        scope_name = "net_" + str(self._thread_index)
        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            self.W_conv1, self.b_conv1 = self._conv_variable(
                [1, 1, FRAME + 1, 64])  # stride=1
            self.W_conv2, self.b_conv2 = self._conv_variable(
                [1, 1, 64, 16])  # stride=1
            self.W_conv3, self.b_conv3 = self._conv_variable(
                [1, 1, 16, 1])  # stride=1

            self.W_fc1, self.b_fc1 = self._fc_variable([2500, 256])

            # lstm
            self.lstm = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)

            # weight for policy output layer
            self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size])

            # weight for value output layer
            self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])

            # state (input)
            self.s = tf.placeholder(
                tf.float32, shape=[None, MAP_SIZE, MAP_SIZE, FRAME + 1], name="state")

            h_conv1 = tf.nn.relu(self._conv2d(self.s, self.W_conv1, 1) + self.b_conv1)
            h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 1) + self.b_conv2)
            h_conv3 = tf.nn.relu(self._conv2d(h_conv2, self.W_conv3, 1) + self.b_conv3)

            h_conv3_flat = tf.reshape(h_conv3, [-1, np.prod(h_conv3.get_shape().as_list()[1:])])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.W_fc1) + self.b_fc1)

            h_fc1_reshape = tf.reshape(h_fc1, [1, -1, 256])  # [1, batch_size, 256]

            # place holder for LSTM unrolling time step size
            self.step_size = tf.placeholder(tf.float32, shape=[1], name="step_size")

            self.initial_lstm_state0 = tf.placeholder(tf.float32, shape=[1, 256], name="ini_lstm_state0")
            self.initial_lstm_state1 = tf.placeholder(tf.float32, shape=[1, 256], name="ini_lstm_state1")
            self.initial_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(self.initial_lstm_state0, self.initial_lstm_state1)
            # Unrolling LSTM up to LOCAL_T_MAX time steps.
            # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
            # Unrolling step size is applied via self.step_size placeholder.
            # When forward propagating, step_size is 1.
            # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
            lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                              h_fc1_reshape,
                                                              initial_state=self.initial_lstm_state,
                                                              sequence_length=self.step_size,
                                                              time_major=False,
                                                              scope=scope)
            # lstm_outputs: (1, sequence_length, 256) for back prop, (1, 1, 256) for
            # forward prop.

            lstm_outputs = tf.reshape(lstm_outputs, [-1, 256])

            # policy (output)
            self.pi = tf.nn.softmax(tf.matmul(lstm_outputs, self.W_fc2) + self.b_fc2)

            # value (output)
            v_ = tf.matmul(lstm_outputs, self.W_fc3) + self.b_fc3
            self.v = tf.reshape(v_, [-1])

            scope.reuse_variables()  # ??
            self.W_lstm = tf.get_variable("BasicLSTMCell/Linear/Matrix")  # scope name??
            self.b_lstm = tf.get_variable("BasicLSTMCell/Linear/Bias")

            self.reset_state()

    def reset_state(self):
        self.lstm_state_out = tf.nn.rnn_cell.LSTMStateTuple(np.zeros([1, 256]), np.zeros([1, 256]))

    def run_policy_and_value(self, sess, s_t, pos=None):
        # used when forward propagating, so step size is 1
        pi_out, v_out, self.lstm_state_out = sess.run([self.pi, self.v, self.lstm_state],
                                                      feed_dict={self.s: [s_t],
                                                                 self.initial_lstm_state0: self.lstm_state_out[0],
                                                                 self.initial_lstm_state1: self.lstm_state_out[1],
                                                                 self.step_size: [1]})
        # pi_out: (1, 3), v_out: (1)
        return pi_out[0], v_out[0]

    def run_policy(self, sess, s_t, pos=None):
        # used for displaying the result with display tool:
        pi_out, self.lstm_state_out = sess.run([self.pi, self.lstm_state],
                                               feed_dict={self.s: [s_t],
                                                          self.initial_lstm_state0: self.lstm_state_out[0],
                                                          self.initial_lstm_state1: self.lstm_state_out[1],
                                                          self.step_size: [1]})
        return pi_out[0]

    def run_value(self, sess, s_t, pos=None):
        # used for calculating V for bootstrapping at the end of LOCAL_T_MAX time step sequence
        # when next sequence starts, V will be calculated again with the same state using updated network weights,
        # so we don't update LSTM state here
        prev_lstm_state_out = self.lstm_state_out
        v_out, _ = sess.run([self.v, self.lstm_state],
                            feed_dict={self.s: [s_t],
                                       self.initial_lstm_state0: self.lstm_state_out[0],
                                       self.initial_lstm_state1: self.lstm_state_out[1],
                                       self.step_size: [1]})
        # roll back lstm state
        self.lstm_state_out = prev_lstm_state_out
        return v_out[0]

    def get_vars(self):
        return [self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_conv3, self.b_conv3,
                self.W_fc1, self.b_fc1,
                self.W_lstm, self.b_lstm,
                self.W_fc2, self.b_fc2,
                self.W_fc3, self.b_fc3
                ]


# Actor-Critic VIN Network
class GameAC_VIN_Network(GameACNetwork):

    def __init__(self, action_size, thread_index, device="/cpu:0"):
        GameACNetwork.__init__(self, action_size, thread_index, device)

        scope_name = "net_" + str(self._thread_index)
        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            k = CONFIG['k']  # Number of value iterations performed
            ch_i = CONFIG['ch_i']  # Channels in input layer
            ch_h = CONFIG['ch_h']  # Channels in initial hidden layer
            ch_q = CONFIG['ch_q']  # Channels in q layer (~actions)
            kern = CONFIG['kern']  # Kernel size of conv

            # input: state, pos
            self.s = tf.placeholder(tf.float32, shape=[None, MAP_SIZE, MAP_SIZE, ch_i], name="state")
            self.pos = tf.placeholder(tf.int32, shape=[None, 2], name="pos")

            # weights from inputs to q layer (~reward in Bellman equation)
            self.w0, self.bias0 = self._conv_variable([kern, kern, ch_i, ch_h], name="w0")
            self.w1, self.bias1 = self._conv_variable([1, 1, ch_h, 1], name="w_r")
            self.w2, self.bias2 = self._conv_variable([kern, kern, 1, ch_q], name="w_q")
            # feedback weights from v layer into q layer (~transition probabilities in
            # Bellman equation)
            self.w3, self.bias3 = self._conv_variable([kern, kern, 1, ch_q], name="w_fb")
            self.w4, self.bias4 = self._fc_variable([ch_q, action_size], name="w_o")
            self.w5, self.bias5 = self._fc_variable([ch_q, 1], name="w_v")

            # initial conv layer over image+reward prior
            h = self._conv2d_flipkernel(self.s, self.w0, name="h0") + self.bias0

            r = self._conv2d_flipkernel(h, self.w1, name="r") + self.bias1
            q = self._conv2d_flipkernel(r, self.w2, name="q") + self.bias2
            v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

            for i in range(0, k - 1):
                rv = tf.concat_v2([r, v], 3)
                wwfb = tf.concat_v2([self.w2, self.w3], 2)

                q = self._conv2d_flipkernel(rv, wwfb, name="q") + self.bias3
                v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")
            # do one last convolution (NHWC)
            q = self._conv2d_flipkernel(tf.concat_v2([r, v], 3),
                                        tf.concat_v2([self.w2, self.w3], 2), name="q") + self.bias3
            # CHANGE TO THEANO ORDERING
            # Since we are selecting over channels, it becomes easier to work with
            # the tensor when it is in NCHW format vs NHWC
            # q = tf.transpose(q, perm=[0, 3, 1, 2])

            # Select the conv-net channels at the state position (S1,S2).
            # This intuitively corresponds to each channel representing an action, and the convnet the Q function.
            # The tricky thing is we want to select the same (S1,S2) position *for each* channel and for each sample
            # TODO: performance can be improved here by substituting expensive
            #       transpose calls with better indexing for gather_nd
            bs = tf.shape(q)[0]
            rprn = tf.reshape(tf.range(bs), [-1, 1])
            idx_in = tf.concat_v2([rprn, self.pos], axis=1)
            q_out = tf.reshape(tf.gather_nd(q, idx_in, name="q_out"), [-1, ch_q])

            # policy (output)
            self.pi = tf.nn.softmax(tf.matmul(q_out, self.w4) + self.bias4)
            # value (output)
            v_ = tf.matmul(q_out, self.w5) + self.bias5
            self.v = tf.reshape(v_, [-1])

    def run_policy_and_value(self, sess, s_t, pos):
        pi_out, v_out = sess.run([self.pi, self.v], feed_dict={self.s: [s_t], self.pos: [pos]})
        return pi_out[0], v_out[0]

    def run_policy(self, sess, s_t, pos):
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t], self.pos: [pos]})
        return pi_out[0]

    def run_value(self, sess, s_t, pos):
        v_out = sess.run(self.v, feed_dict={self.s: [s_t], self.pos: [pos]})
        return v_out[0]

    def get_vars(self):
        return [self.w0, self.bias0,
                self.w1, self.bias1,
                self.w2, self.bias2,
                self.w3, self.bias3,
                self.w4, self.bias4,
                self.w5, self.bias5
                ]
