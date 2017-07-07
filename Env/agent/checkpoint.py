import random
from base import Base

CHECKPOINT_COLOR = (64, 64, 64)


class Checkpoint(Base):

    def __init__(
            self,
            identity,
            pos_x, pos_y,
            size=1, color=CHECKPOINT_COLOR
    ):
        super(Checkpoint, self).__init__(identity, pos_x, pos_y, size, color)

    def step_back(self):
        pass

    def collide_walls(self, screen_rect):
        return False

    def switch_components(self, other, i):
        print "ch"
        pass

    def collide(self, others):
        pass

    def update(self, screen_rect, others, info):
        super(Checkpoint, self).update(screen_rect, others, None)

    @classmethod
    def random(cls, map_size, size):

        pos_x = random.randint(0 + size / 2, map_size - size / 2 - 1)
        pos_y = random.randint(0 + size / 2, map_size - size / 2 - 1)
        return pos_x, pos_y
