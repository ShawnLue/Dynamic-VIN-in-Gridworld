import random
from base import Base, ACTION_LIST
from ..utils.utils import gen_margin
AGENT_COLOR = (0, 0, 0)


class Agent(Base):

    def __init__(
            self,
            identity,
            pos_x, pos_y,
            size=1, color=AGENT_COLOR
    ):
        super(Agent, self).__init__(identity, pos_x, pos_y, size, color)

    def set_cmd(self, num):
        self.vector[0] = ACTION_LIST[num][0]
        self.vector[1] = ACTION_LIST[num][1]
    # advanced functions of an agent

    def step_back(self):
        super(Agent, self).step_back()

    def collide_walls(self, screen_rect):
        return super(Agent, self).collide_walls(screen_rect)

    def switch_components(self, other, i):
        super(Agent, self).switch_components(other, i)

    def collide(self, others):
        return super(Agent, self).collide(others)

    def update(self, screen_rect, others, info):
        super(Agent, self).update(screen_rect, others, info)
        # print "in update:", str(self.vector)
        # print "pos", str(self.true_pos)
        # # exit()

    @classmethod
    def random(cls, map_size, size, difficulty=None, ch_pos=None):
        if difficulty is None:
            pos_x = random.randint(0 + size / 2, map_size - size / 2 - 1)
            pos_y = random.randint(0 + size / 2, map_size - size / 2 - 1)
        else:
            # assert ch_pos is not None
            candidate_pos_list = gen_margin(map_size, size, difficulty, ch_pos)
            if len(candidate_pos_list) != 0:
                pos_x, pos_y = random.choice(candidate_pos_list)
            else:
                pos_x = random.randint(0 + size / 2, map_size - size / 2 - 1)
                pos_y = random.randint(0 + size / 2, map_size - size / 2 - 1)

        return pos_x, pos_y
