# VIN settings
GOAL_OUT_FLAG = 1.0
OB_OUT_FLAG = 1.0
ACTION_SIZE = 9  # action size
FRAME = 1

SUB_SEQ_DIM = 5

CONFIG = {'k': 40,  # Number of value iterations performed
          'ch_i': FRAME+1,  # Channels in input layer
          'ch_h': 250,  # Channels in initial hidden layer
          'ch_q': 20,  # Channels in q layer (~actions)
          'kern': 7  # Kernel size of conv
          }

USE_GPU = True
NUM_GLOBAL_STEPS = 30000000
AUX_REWARD = False

#  Pygame GridWorld Environment
# -----------
# 1. map size(square map)
MAP_SIZE = 50
# 2. the number of agent/obstacle/checkpoint(goal)
MAX_OB = 10
MAX_DIFFICULTY = 30
INPUT_DICT = {'ag': 1, 'ob': 0, 'ch': 1}
# 3. save the screenshot or not (the screenshot is saved in
# ./output_data/image/ if it is set to be True)
SCREEN_CAPTURE = False
# 4. quit the game if collision happens
COLLISION_QUIT = False
# 5. Static
STATIC = True
