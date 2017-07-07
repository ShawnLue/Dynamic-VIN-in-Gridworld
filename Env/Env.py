import pygame as pg
import numpy as np
import os
import random
import copy

from agent.base import ACTION_LIST
from agent.agent import Agent
from agent.checkpoint import Checkpoint
from agent.obstacle import Obstacle
from utils.utils import act_cost

IMAGE_DIR = os.path.abspath('.') + '/output_data/image/'

CAPTION = "RL-Planning@Horizon-Robotics"
BACKGROUND_COLOR = (255, 255, 255)

AGENT_SIZE = 1
CHECKPOINT_SIZE = 1
OB_SCALE_SIZE_LIST = [1, 3, 3, 5, 5, 7, 7]
# OB_SCALE_SIZE_LIST = [5, 9, 13, 17]
# OB_SCALE_SIZE_LIST = [51, 81, 111, 141]


# FLAGS OF ELEMENTS IN ENV
OBSTACLE_FLAG = 1
SPACE_FLAG = 0
GOAL_FLAG = 10
AGENT_POS_FLAG = 100


def buffer_filter(buff):
    buff[abs(buff - 64) < 30] = GOAL_FLAG  # Goal
    buff[abs(buff - 128) < 30] = OBSTACLE_FLAG  # Obstacle
    buff[abs(buff - 0) == 0] = AGENT_POS_FLAG  # Agent
    buff[abs(buff - 255) == 0] = SPACE_FLAG  # Space
    return buff


class Env:

    def __init__(self, map_size=50, snapshot=False, collision_quit=False):
        '''
        map_size:
        snapshot: (unresolved)
        collision_quit:
        info: env information for remaking
        '''
        self.map_size = map_size
        self.snapshot = snapshot
        self.collision_quit = collision_quit

        self.step_counter = 0

        self.blocks = pg.sprite.Group()

        pg.init()
        os.environ["SDL_VIDEO_CENTERED"] = "True"
        pg.display.set_caption(CAPTION)
        self.screen = pg.display.set_mode((self.map_size, self.map_size), 0, 8)
        self.screen_rect = self.screen.get_rect()

        self.agent_num = 0
        self.ob_num = 0
        self.check_num = 0
        self.done = False

        try:
            os.makedirs(IMAGE_DIR)
        except:
            pass

    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True
        keys = pg.key.get_pressed()
        if keys[pg.K_r]:  # Reset: press r
            self.done = True
        if keys[pg.K_q]:  # Quit: press q
            exit()

    def get_state(self):
        # Img
        buff = pg.image.tostring(self.screen, "P")
        buff = copy.deepcopy(buff)
        buff = np.fromstring(buff, dtype=np.uint8)
        buff = buff.reshape(self.map_size, self.map_size)
        buff = buffer_filter(buff)
        return buff

    @classmethod
    def random_scene(cls, map_size, input_data, difficulty=None):
        agent_num = input_data['ag']
        ob_num = input_data['ob']
        check_num = input_data['ch']

        blocks = pg.sprite.Group()
        blocks.empty()

        element_dict = {'ag': [], 'ob': [], 'ch': []}

        # For obstacle generation:
        for i in range(ob_num):
            while True:
                i_size = random.choice(OB_SCALE_SIZE_LIST)
                pos_x, pos_y, tmp_vector = Obstacle.random(map_size, i_size)
                tmp = Obstacle(i, pos_x, pos_y, i_size, tmp_vector)
                collision = False
                for j in blocks:
                    if pg.sprite.collide_rect(tmp, j):
                        collision = True
                        break
                if not collision:
                    blocks.add(tmp)
                    element_dict['ob'].append([i, pos_x, pos_y, i_size, tmp_vector])
                    break

        # For checkpoint generation:
        for i in range(check_num):
            while True:
                pos_x, pos_y = Checkpoint.random(
                    map_size, CHECKPOINT_SIZE)
                tmp = Checkpoint(i, pos_x, pos_y, CHECKPOINT_SIZE)
                collision = False
                for j in blocks:
                    if pg.sprite.collide_rect(tmp, j):
                        collision = True
                        break
                if not collision:
                    blocks.add(tmp)
                    element_dict['ch'].append([i, pos_x, pos_y, CHECKPOINT_SIZE])
                    break

        # For agent generation:
        for i in range(agent_num):
            ch_pos = (difficulty is not None) and (element_dict['ch'][0][1], element_dict['ch'][0][2]) or None
            while True:
                pos_x, pos_y = Agent.random(map_size, AGENT_SIZE, difficulty, ch_pos)
                tmp = Agent(i, pos_x, pos_y, AGENT_SIZE)
                collision = False
                for j in blocks:
                    if pg.sprite.collide_rect(tmp, j):
                        collision = True
                        break
                if not collision:
                    blocks.add(tmp)
                    element_dict['ag'].append([i, pos_x, pos_y, AGENT_SIZE])
                    break

        return element_dict

    def reset(self, input_dict, static=False):
        self.step_counter = 0
        self.done = False
        if input_dict is None:
            raise ValueError('input_dict is None.')

        self.blocks.empty()
        self.agent_num = len(input_dict['ag'])
        self.ob_num = len(input_dict['ob'])
        self.check_num = len(input_dict['ch'])

        # For obstacle generation:
        for i in input_dict['ob']:
            if static:
                i[-1] = (0, 0)
            self.blocks.add(Obstacle(*i))
        for i in input_dict['ag']:
            self.blocks.add(Agent(*i))
        for i in input_dict['ch']:
            self.blocks.add(Checkpoint(*i))

        self.draw()
        self.event_loop()
        pg.display.update()
        # pg.time.delay(20)

        # TODO
        if self.snapshot:
            pass
            # TODO: snapshot
        state = self.get_state()
        return state

    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.blocks.draw(self.screen)

    def step(self, cmds):
        assert isinstance(cmds, list) and len(cmds) == self.agent_num
        self.step_counter += 1
        # print "step_counter: " + str(self.step_counter)

        for e in self.blocks:
            if type(e).__name__ == 'Agent':
                e.set_cmd(cmds[e.identity])
        info = {}
        self.event_loop()
        self.blocks.update(self.screen_rect, self.blocks, info)
        self.draw()
        pg.display.update()
        # pg.time.delay(20)

        # snapshot
        if self.snapshot:
            pass
            # TODO: snapshot
        state = self.get_state()
        reward = 0.0
        if self.agent_num != 0:
            reward = -act_cost(cmds[0]) - 1.0 * info['wallCrash'] -\
                     2.0 * len(info['obCrash']) -\
                     1.0 * len(info['agCrash']) + 1.0 * info['chCrash']
            if info['chCrash'] or (self.collision_quit and (
                        len(info['obCrash']) + len(info['agCrash'])) > 0):
                self.done = True

        return state, reward, self.done, info
