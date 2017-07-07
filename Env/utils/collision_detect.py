# ----------------------------------------------------
# Second edition of HAADS(used for RL simulation)
# Author: Xiangyu Liu
# Date: 2016.11.26
# Filename: collision_detect.py
# ----------------------------------------------------

import pygame as pg


def collide_other(one, two):
    """
    Callback function for use with pg.sprite.collidesprite methods.
    It simply allows a sprite to test collision against its own group,
    without returning false positives with itself.
    """
    return one is not two and pg.sprite.collide_rect(one, two)
