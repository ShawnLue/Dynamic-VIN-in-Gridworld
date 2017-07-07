import numpy as np
import copy

from Env.Env import OBSTACLE_FLAG, AGENT_POS_FLAG, GOAL_FLAG, SPACE_FLAG
from constants import MAP_SIZE, GOAL_OUT_FLAG, OB_OUT_FLAG, FRAME


def process_state(state, need_goal=True):
    """
    Return a single input to the VIN network
    :param state: img from the game
    :param need_goal: goal
    :return:
    """
    state = state.reshape(MAP_SIZE, MAP_SIZE, 1)
    # TODO:
    im_data = state.astype('float32')
    GOAL_POS = np.where(im_data == GOAL_FLAG)
    AGENT_POS = np.where(im_data == AGENT_POS_FLAG)

    im_data[GOAL_POS] = 0.0
    im_data[AGENT_POS] = 0.0
    im_data[np.where(im_data == OBSTACLE_FLAG)] = OB_OUT_FLAG

    pos = [AGENT_POS[0][0], AGENT_POS[1][0]]

    if need_goal:
        value_data = np.zeros_like(im_data).astype('float32')
        value_data[GOAL_POS] = GOAL_OUT_FLAG
        return im_data, value_data, pos
    else:
        return im_data, pos


def get_subsequence(env, map_info, frame_count):
    map_info_tmp = copy.deepcopy(map_info)
    map_info_tmp['ag'] = []

    # get goal map
    state_org = np.expand_dims(env.reset(map_info_tmp, False), -1)
    GOAL_POS = np.where(state_org == GOAL_FLAG)
    value_data = np.zeros_like(state_org).astype('float32')
    value_data[GOAL_POS] = GOAL_OUT_FLAG
    state_org[GOAL_POS] = 0.0
    # create subsequence
    result = np.empty([frame_count, MAP_SIZE, MAP_SIZE, FRAME + 1])
    for i in range(frame_count):
        tmp = np.expand_dims(env.step([])[0], -1)
        tmp[GOAL_POS] = 0.0
        result[i] = np.concatenate([tmp, value_data], axis=-1)
    return result
