"""Defines an agent that learns to play Atari games using an A3C architecture.
Heavily influenced by DeepMind's seminal paper 'Asynchronous Methods for Deep Reinforcement
Learning' (Mnih et al., 2016).
"""

import a3c
import logging
import numpy as np
import time
import tensorflow as tf
from Env.Env import Env
import os

from scipy import signal
from utils import process_state, get_subsequence
from constants import MAP_SIZE, INPUT_DICT, STATIC, FRAME, USE_GPU, CONFIG,\
    NUM_GLOBAL_STEPS, MAX_OB, MAX_DIFFICULTY, AUX_REWARD, SUB_SEQ_DIM

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def _apply_discount(rewards, discount):
    """Discounts the specified rewards exponentially.
    Given rewards = [r0, r1, r2, r3] and discount = 0.99, the result is:
        [r0 + 0.99 * (r1 + 0.99 * (r2 + 0.99 * r3)),
         r1 + 0.99 * (r2 + 0.99 * r3),
         r2 + 0.99 * r3,
         r3]
    Example: rewards = [10, 20, 30, 40] and discount = 0.99 -> [98.01496, 88.904, 69.6, 40].
    Returns:
        The discounted rewards.
    """

    return signal.lfilter([1], [1, -discount], rewards[::-1])[::-1]


class Agent:
    def __init__(self,
                 worker_index,
                 env,
                 render,
                 num_local_steps,
                 learning_rate,
                 entropy_regularization,
                 max_gradient_norm,
                 discount,
                 summary_writer,
                 summary_update_interval):
        """An agent that learns to plan in gridworld using an A3C architecture.
        Args:
            worker_index: Index of the worker thread that is running this agent.
            env: A simulator object (see in /Env') that wraps over a pygame environment.
            render: Determines whether to display the game screen.
            num_local_steps: Number of experiences used per worker when updating the model.
            learning_rate: The speed with which the network learns from new examples.
            entropy_regularization: The strength of the entropy regularization term.
            max_gradient_norm: Maximum value allowed for the L2-norms of gradients. Gradients with
                norms that would otherwise surpass this value are scaled down. ?
            discount: Discount factor for future rewards.
            summary_writer: A TensorFlow object that writes summaries.
            summary_update_interval: Number of training steps needed to update the summary data.
        """

        self.worker_index = worker_index
        self.env = env
        self.render = render
        self.num_local_steps = num_local_steps
        self.discount = discount
        self.summary_writer = summary_writer
        self.summary_update_interval = summary_update_interval
        self.num_times_trained = 0

        device = "cpu"
        if USE_GPU:
            device = "gpu"
        worker_device = '/job:thread/task:{}/{}:0'.format(worker_index, device)

        # Get global parameters
        with tf.device(tf.train.replica_device_setter(1, '/job:master', worker_device)):
            # ps_tasks, ps_device, worker_device
            with tf.variable_scope('global'):
                self.global_network = a3c.PolicyNetwork()
                self.global_step = tf.get_variable('global_step', [],
                                                   tf.int32, tf.constant_initializer(0, tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope('local'):
                self.local_network = a3c.PolicyNetwork()
                self.local_network.global_step = self.global_step

            self.action = tf.placeholder(tf.int32, [None], 'Action')
            self.advantage = tf.placeholder(tf.float32, [None], 'Advantage')
            self.discounted_reward = tf.placeholder(tf.float32, [None], 'Discounted_Reward')

            # Estimate the policy loss using the cross-entropy loss function.
            action_logits = self.local_network.action_logits
            # policy_loss part I: policy gradient
            policy_loss = tf.reduce_sum(
                self.advantage * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_logits,
                                                                                labels=self.action)
            )
            # Regularize the policy loss by adding uncertainty (subtracting entropy). High entropy means
            # the agent is uncertain (meaning, it assigns similar probabilities to multiple actions).
            # Low entropy means the agent is sure of which action it should perform next.
            entropy = -tf.reduce_sum(tf.nn.softmax(action_logits) * tf.nn.log_softmax(action_logits))
            # policy_loss part II: entropy loss
            policy_loss -= entropy_regularization * entropy

            # Estimate the value loss using the sum of squared errors.
            value_loss = tf.nn.l2_loss(self.local_network.value - self.discounted_reward)

            # Estimate the final loss.
            self.loss = policy_loss + 0.5 * value_loss

            # Fetch and clip the gradients of the local network.
            gradients = tf.gradients(self.loss, self.local_network.parameters)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

            # Update the global network using the clipped gradients.
            batch_size = tf.shape(self.local_network.s)[0]
            grads_and_vars = list(zip(clipped_gradients, self.global_network.parameters))
            self.train_step = [tf.train.AdamOptimizer(learning_rate).apply_gradients(grads_and_vars),
                               self.global_step.assign_add(batch_size)]

            # Synchronize the local network with the global network.
            self.reset_local_network = [local_p.assign(global_p)
                                        for local_p, global_p in zip(self.local_network.parameters,
                                                                     self.global_network.parameters)]

            tf.summary.scalar('model/loss', self.loss / tf.to_float(batch_size))
            tf.summary.scalar('model/policy_loss', policy_loss / tf.to_float(batch_size))
            tf.summary.scalar('model/value_loss', value_loss / tf.to_float(batch_size))
            tf.summary.scalar('model/entropy', entropy / tf.to_float(batch_size))
            tf.summary.scalar('model/global_norm', tf.global_norm(self.local_network.parameters))
            tf.summary.scalar('model/gradient_global_norm', tf.global_norm(gradients))
            self.summary_step = tf.summary.merge_all()

    def _get_experiences(self, difficulty, ob):
        states = []
        poses = []
        actions = []
        rewards = []
        values = []
        subsequences = []

        if not self.render:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        # ob_num = int(round(np.random.normal(MAX_OB * round(float(self.global_step.eval()) / NUM_GLOBAL_STEPS), 1)))
        INPUT_DICT['ob'] = ob if ob >= 0 else 0
        difficulty = (difficulty < 1) and 1 or difficulty

        map_info = Env.random_scene(MAP_SIZE, INPUT_DICT, difficulty=difficulty)

        subsequence = get_subsequence(self.env, map_info, self.num_local_steps + SUB_SEQ_DIM)

        state_org = self.env.reset(map_info, STATIC)
        state, s_t_goal, pos = process_state(state_org, need_goal=True)
        state = np.concatenate((state, ) * FRAME, axis=-1)
        state = np.concatenate([state, s_t_goal], axis=-1)
        # record summary
        episode_reward = 0.0
        episode_length = 0
        tstart = time.time()
        terminal = False

        print ""
        for ind in range(self.num_local_steps):
            print "step: " + str(ind) + " ob: " + str(INPUT_DICT['ob']) + " difficulty: " + str(difficulty)
            # reward = 0.0
            action, value = self.local_network.sample_action(state, pos, subsequence[ind:ind+SUB_SEQ_DIM])
            '''
            if AUX_REWARD:
                try:
                    instruct_action = AStarBlock(state_org)['act_seq'][0]
                except (KeyError, NotImplementedError, RuntimeError):
                    instruct_action = 0
                reward += reward_mapping(action, instruct_action)
            '''
            state_org, reward_tmp, terminal, _ = self.env.step([action])
            # reward += reward_tmp

            # Store this experience.
            states.append(state)
            poses.append(pos)
            actions.append(action)
            # rewards.append(reward)
            rewards.append(reward_tmp)
            values.append(value)
            subsequences.append(subsequence[ind:ind+SUB_SEQ_DIM])

            s_t2, pos = process_state(state_org, need_goal=False)

            state = np.insert(state, -1, s_t2.squeeze(), axis=-1)[:, :, 1:]

            episode_reward += reward_tmp
            episode_length += 1

            if terminal:
                print "Wonderful Path!"
                break

        run_time = time.time() - tstart
        LOGGER.info('Finished episode. Total reward: %d. Length: %d.',
                    episode_reward, episode_length)
        summary = tf.Summary()
        summary.value.add(tag='environment/episode_length',
                          simple_value=episode_length)
        summary.value.add(tag='environment/episode_reward',
                          simple_value=episode_reward)
        summary.value.add(tag='environment/fps',
                          simple_value=episode_length / run_time)

        self.summary_writer.add_summary(summary, self.global_step.eval())
        self.summary_writer.flush()

        # Estimate discounted rewards.
        rewards = np.array(rewards)
        next_value = 0 if terminal else self.local_network.estimate_value(state, pos, subsequence[ind:ind+SUB_SEQ_DIM])
        discounted_rewards = _apply_discount(np.append(rewards, next_value), self.discount)[:-1]

        # Estimate advantages.
        values = np.array(values + [next_value])
        advantages = _apply_discount(rewards + self.discount * values[1:] - values[:-1],
                                     self.discount)
        return np.array(states), np.array(poses), np.array(actions), advantages,\
               discounted_rewards, sum(rewards), np.array(subsequences)

    def train(self, sess, difficulty, ob):

        """Performs a single learning step.
        Args:
            :param sess:
            :param difficulty: distance for obstacles
        """

        sess.run(self.reset_local_network)
        states, poses, actions, advantages, discounted_rewards, episode_reward, subsequences =\
            self._get_experiences(difficulty, ob)
        feed_dict = {self.local_network.s: states,
                     self.local_network.pos: poses,
                     self.local_network.subsequence: subsequences,
                     self.action: actions,
                     self.advantage: advantages,
                     self.discounted_reward: discounted_rewards}

        # Occasionally write summaries.
        # Only worker 0 write summaries(self.summary_step)
        if self.num_times_trained % self.summary_update_interval == 0:
            _, global_step, summary = sess.run(
                [self.train_step, self.global_step, self.summary_step], feed_dict)
            self.summary_writer.add_summary(tf.Summary.FromString(summary), global_step)
            self.summary_writer.flush()
        else:
            _, global_step = sess.run([self.train_step, self.global_step], feed_dict)

        self.num_times_trained += 1

        return global_step, episode_reward
