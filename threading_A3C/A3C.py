import tensorflow as tf
import threading

import signal
import math
import os
import time

from AC_Network import GameAC_LSTM_Network, GameAC_FF_Network, GameAC_VIN_Network
from A3C_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

from constants import USE_GPU, USE_NET, ACTION_SIZE
from constants import INITIAL_ALPHA_HIGH, INITIAL_ALPHA_LOG_RATE, INITIAL_ALPHA_LOW
from constants import RMSP_ALPHA, RMSP_EPSILON, GRAD_NORM_CLIP
from constants import PARALLEL_SIZE, MAX_TIME_STEP

from constants import LOG_FILE, CHECKPOINT_DIR


def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)


device = "/cpu:0"
if USE_GPU:
    device = "/gpu:0"

initial_lr = log_uniform(INITIAL_ALPHA_LOW, INITIAL_ALPHA_HIGH, INITIAL_ALPHA_LOG_RATE)

global_t = 0

stop_requested = False

if USE_NET == "FF":
    global_network = GameAC_FF_Network(ACTION_SIZE, -1, device)
elif USE_NET == "LSTM":
    global_network = GameAC_LSTM_Network(ACTION_SIZE, -1, device)
else:  # VIN
    global_network = GameAC_VIN_Network(ACTION_SIZE, -1, device)

training_threads = []

lr_input = tf.placeholder(tf.float32, name="lr_input")

grad_applier = RMSPropApplier(learning_rate=lr_input,
                              decay=RMSP_ALPHA,
                              momentum=0.0,
                              epsilon=RMSP_EPSILON,
                              clip_norm=GRAD_NORM_CLIP,
                              device=device)

for i in range(PARALLEL_SIZE):
    training_thread = A3CTrainingThread(i, global_network, initial_lr,
                                        lr_input, grad_applier, MAX_TIME_STEP, device=device)
    training_threads.append(training_thread)

config_tf = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config_tf.gpu_options.allow_growth = True
sess = tf.Session(config=config_tf)

init = tf.global_variables_initializer()
sess.run(init)

# summary for tensorboard
# TODO: Add loss info to summary
score_input = tf.placeholder(tf.int32, name="score_input")
tf.summary.scalar("score", score_input)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

# init or load checkpoint with Saver
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print "checkpoint loaded:", checkpoint.model_checkpoint_path
    tokens = checkpoint.model_checkpoint_path.split("-")
    # set global step
    global_t = int(tokens[1])
    print ">>> global step set: ", global_t
    # set wall time
    wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
    with open(wall_t_fname, 'r') as f:
        wall_t = float(f.read())
else:
    print "Could not find old checkpoint"
    # set wall time
    wall_t = 0.0


############################################function##############################################

def train_function(parallel_index):
    global global_t
    training_thread = training_threads[parallel_index]
    # set start_time
    start_time = time.time() - wall_t
    training_thread.set_start_time(start_time)

    while True:
        if stop_requested:
            break
        if global_t > MAX_TIME_STEP:
            break
        diff_global_t = training_thread.process(sess, global_t, summary_writer, summary_op, score_input)
        global_t += diff_global_t


def signal_handler(signal, frame):
    global stop_requested
    print 'You pressed Ctrl+C!'
    stop_requested = True

#####################################################################################################

process_threads = []
for i in range(PARALLEL_SIZE):
    process_threads.append(threading.Thread(target=train_function, args=(i, )))

signal.signal(signal.SIGINT, signal_handler)

# set start time
start_time = time.time() - wall_t

for t in process_threads:
    t.start()

print 'Press Ctrl+C to stop'
signal.pause()

print 'Now saving data. Please wait'

for t in process_threads:
    t.join()

if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

# write wall time
wall_t = time.time() - start_time
wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
with open(wall_t_fname, 'w') as f:
    f.write(str(wall_t))
saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step=global_t)