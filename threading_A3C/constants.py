# Local thread
LOCAL_T_MAX = 10  # repeat step size

# A3C Log settings
LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000
CHECKPOINT_DIR = './tmp/checkpoints'
LOG_FILE = './tmp/a3c_log'

# Optimizer
RMSP_ALPHA = 0.99  # decay parameter for RMSProp
RMSP_EPSILON = 1e-6  # epsilon parameter for RMSProp
GRAD_NORM_CLIP = 40.0  # gradient norm clipping

# Learning rate
INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate
INITIAL_ALPHA_LOG_RATE = 0.4226  # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
LR_MINIMUM = 1e-5  # minimum learning rate

# Parallel
PARALLEL_SIZE = 8  # parallel thread size

ACTION_SIZE = 9  # action size
FRAME = 4

GAMMA = 0.99  # discount factor for rewards
ENTROPY_BETA = 0.01  # entropy regularization constant
MAX_TIME_STEP = 10 * 10**7

USE_GPU = True  # To use GPU, set True
USE_NET = "VIN"  # A3C_FF / A3C_LSTM / A3C_VIN

# VIN settings
GOAL_OUT_FLAG = 1.0
OB_OUT_FLAG = 1.0

CONFIG = {'k': 60,  # Number of value iterations performed
          'ch_i': FRAME+1,  # Channels in input layer
          'ch_h': 200,  # Channels in initial hidden layer
          'ch_q': 20,  # Channels in q layer (~actions)
          'kern': 5  # Kernel size of conv
          }



#  Pygame GridWorld Environment
# -----------
# 1. map size(square map)
MAP_SIZE = 50
# 2. the number of agent/obstacle/checkpoint(goal)
INPUT_DICT = {'ag': 1, 'ob': 40, 'ch': 1}
# 3. save the screenshot or not (the screenshot is saved in
# ./output_data/image/ if it is set to be True)
SCREEN_CAPTURE = False
# 4. quit the game if collision happens
COLLISION_QUIT = False
# 5. Static
STATIC = False