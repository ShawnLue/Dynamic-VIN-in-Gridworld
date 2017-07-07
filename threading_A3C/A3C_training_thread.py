import tensorflow as tf
import numpy as np
import time

from Env.Env import Env

from AC_Network import GameAC_FF_Network, GameAC_LSTM_Network, GameAC_VIN_Network

# Network parameters
from constants import USE_NET, ACTION_SIZE, ENTROPY_BETA
from constants import LOCAL_T_MAX
from constants import LR_MINIMUM
# RL-A3C settings
from constants import FRAME
from constants import GAMMA
from constants import PERFORMANCE_LOG_INTERVAL
# Simulator settings
from constants import MAP_SIZE, SCREEN_CAPTURE, COLLISION_QUIT, INPUT_DICT, STATIC

# Processing the state
from game_state import process_state


class A3CTrainingThread(object):
    def __init__(self, thread_index, global_network,
                 initial_lr, lr_input, grad_applier, max_global_time_step, device):
        self.thread_index = thread_index
        self.lr_input = lr_input
        self.max_global_time_step = max_global_time_step

        if USE_NET == "FF":
            self.local_net = GameAC_FF_Network(ACTION_SIZE, thread_index, device)
        elif USE_NET == "LSTM":
            self.local_net = GameAC_LSTM_Network(ACTION_SIZE, thread_index, device)
        else:  # VIN
            self.local_net = GameAC_VIN_Network(ACTION_SIZE, thread_index, device)
        self.local_net.prepare_loss(ENTROPY_BETA)

        self.game = Env(MAP_SIZE, SCREEN_CAPTURE, COLLISION_QUIT)

        with tf.device(device):
            var_refs = [v._ref() for v in self.local_net.get_vars()]
            self.gradients = tf.gradients(self.local_net.total_loss, var_refs,
                                          gate_gradients=False,
                                          aggregation_method=None,
                                          colocate_gradients_with_ops=False)
            # gradients only updated upon global_net
        self.apply_gradients = grad_applier.apply_gradients(global_network.get_vars(),
                                                            self.gradients)
        self.sync = self.local_net.sync_from(global_network)

        self.local_t = 0
        self.initial_lr = initial_lr
        self.episode_reward = 0

        # variable controlling log output
        self.prev_local_t = 0

    def _anneal_lr(self, global_time_step):
        lr = self.initial_lr * float(self.max_global_time_step - global_time_step) / self.max_global_time_step
        if lr < LR_MINIMUM:
            lr = LR_MINIMUM
        return lr

    # A3C is an on-policy algorithm
    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
        summary_str = sess.run(summary_op, feed_dict={score_input: score})
        summary_writer.add_summary(summary_str, global_t)
        summary_writer.flush()

    def set_start_time(self, start_time):
        self.start_time = start_time

    def process(self, sess, global_t, summary_writer, summary_op, score_input):
        # move <= LOCAL_MAX_T steps(or get done) & update & over
        states = []
        poses = []
        actions = []
        rewards = []
        values = []
        terminal_end = False

        # copy weights from shared to local
        sess.run(self.sync)

        start_local_t = self.local_t

        if USE_NET == "LSTM":
            start_lstm_state = self.local_net.lstm_state_out

        # t_max times loop
        map_info_thread = Env.random_scene(MAP_SIZE, INPUT_DICT)

        s_t_single = self.game.reset(map_info_thread, STATIC)
        s_t_single, s_t_goal, pos = process_state(s_t_single, need_goal=True)

        s_t = np.concatenate((s_t_single,) * FRAME, axis=-1)
        s_t = np.concatenate([s_t, s_t_goal], axis=-1)

        for i in range(LOCAL_T_MAX):
            pi_, value_ = self.local_net.run_policy_and_value(sess, s_t, pos)
            action = self.choose_action(pi_)

            states.append(s_t)
            poses.append(pos)
            actions.append(action)
            values.append(value_)

            # process game
            s_t2_single, r_thread, done_thread, _ = self.game.step([action])
            s_t2_single, pos = process_state(s_t2_single, need_goal=False)

            self.episode_reward += r_thread
            # clip reward
            rewards.append(np.clip(r_thread, -1, 1))

            self.local_t += 1
            print self.local_t

            # s_t2 = np.append(s_t[:, :, 1:], s_t2_single, axis=2)
            s_t2 = np.insert(s_t, -1, s_t2_single.squeeze(), axis=-1)[:, :, 1:]

            s_t = s_t2

            if done_thread:
                terminal_end = True
                print "score={}".format(self.episode_reward)

                self._record_score(sess, summary_writer, summary_op, score_input,
                                   self.episode_reward, global_t)
                self.episode_reward = 0
                # Reset
                map_info_thread = Env.random_scene(MAP_SIZE, INPUT_DICT)
                s_t_single = self.game.reset(map_info_thread, STATIC)
                s_t_single, s_t_goal, pos = process_state(s_t_single, need_goal=True)
                s_t = np.concatenate((s_t_single,) * FRAME, axis=-1)
                s_t = np.concatenate([s_t, s_t_goal], axis=-1)

                if USE_NET == "LSTM":
                    self.local_net.reset_state()
                break
        R = 0.0
        if not terminal_end:
            R = self.local_net.run_value(sess, s_t, pos)
        actions.reverse()
        states.reverse()
        poses.reverse()
        rewards.reverse()
        values.reverse()

        batch_si, batch_pos, batch_a, batch_td, batch_R = [], [], [], [], []

        # compute and accumulate gradients
        for (ai, ri, si, pi, vi) in zip(actions, rewards, states, poses, values):
            R = ri + GAMMA * R
            td = R - vi
            a = np.zeros([ACTION_SIZE])  # one-hot
            a[ai] = 1

            batch_si.append(si)
            batch_pos.append(pi)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        cur_lr = self._anneal_lr(global_t)

        if USE_NET == "LSTM":
            batch_si.reverse()
            batch_a.reverse()
            batch_td.reverse()
            batch_R.reverse()
            sess.run(self.apply_gradients,
                     feed_dict={
                         self.local_net.s: batch_si,
                         self.local_net.pos: batch_pos,
                         self.local_net.a: batch_a,
                         self.local_net.td: batch_td,
                         self.local_net.r: batch_R,
                         self.local_net.initial_lstm_state: start_lstm_state,
                         self.local_net.step_size: [len(batch_a)],
                         self.lr_input: cur_lr}
                     )
        else:
            sess.run(self.apply_gradients,
                     feed_dict={
                         self.local_net.s: batch_si,
                         self.local_net.pos: batch_pos,
                         self.local_net.a: batch_a,
                         self.local_net.td: batch_td,
                         self.local_net.r: batch_R,
                         self.lr_input: cur_lr
                     })
        # TODO: why thread_index == 0?
        if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
            self.prev_local_t += PERFORMANCE_LOG_INTERVAL
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            print "### Performance: {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                global_t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.
            )

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t
