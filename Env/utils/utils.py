# ----------------------------------------------------
# Second edition of HAADS(used for RL simulation)
# Author: Xiangyu Liu
# Date: 2016.11.26
# Filename: utils.py
# ----------------------------------------------------

import numpy as np
import itertools
import math

def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim == 0
        x = x.item()
    if isinstance(x, float):
        rep = "%g" % x
    else:
        rep = str(x)
    return " " * (l - len(rep)) + rep


def fmt_row(width, row):
    out = " | ".join(fmt_item(x, width) for x in row)
    return out


def act_cost(a):
    return (a % 2 == 0 and a != 0) and 0.01 * math.sqrt(2) or 0.01


def gen_margin(map_size, size, diff, pos):
    def legal_map(m_size, pos):
        if 0 <= pos[0] < m_size and 0 <= pos[1] < m_size:
            return True
        else:
            return False

    xmin, xmax = pos[0] - diff, pos[0] + diff
    ymin, ymax = pos[1] - diff, pos[1] + diff
    result = []
    result += list(itertools.product(range(xmin, xmin+1), range(ymin, ymax+1)))
    result += list(itertools.product(range(xmax, xmax+1), range(ymin, ymax+1)))
    result += list(itertools.product(range(xmin, xmax+1), range(ymin, ymin+1)))
    result += list(itertools.product(range(xmin, xmax+1), range(ymax, ymax+1)))
    result = list(set(result))
    result = [item for item in result if legal_map(map_size, item)]
    return result


def reward_mapping(act, instruct):
    if act == 0:
        return -0.1
    if abs(act - instruct) == 4:
        return -0.5
    elif act == instruct:
        print "correct!", act, instruct
        return 0.5
    elif abs(act - instruct) == 1 or abs(act - instruct) == 7:
        return 0.1
    else:
        return -0.1
