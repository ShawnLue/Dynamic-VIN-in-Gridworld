import pygame as pg
from ..utils.collision_detect import collide_other

ACTION_LIST = [(0, 0), (0, -1), (1, -1), (1, 0),
               (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
"""
Action Graph:
8   1   2
7   0   3
6   5   4
"""


class Base(pg.sprite.Sprite):

    def __init__(
            self,
            identity,
            pos_x, pos_y,
            size, color
    ):
        pg.sprite.Sprite.__init__(self)
        self.image = pg.Surface((size, size)).convert()
        self.image.fill(color)
        self.rect = self.image.get_rect(center=(pos_x, pos_y))
        self.true_pos = list(self.rect.center)

        self.identity = identity
        self.size = size
        self.vector = [0, 0]

    def step_back(self):
        """Decrement block's position by one unit pixel."""
        self.true_pos[0] -= self.vector[0]
        self.true_pos[1] -= self.vector[1]
        self.rect.center = self.true_pos

    def collide_walls(self, screen_rect):
        """
        Reverse relevent velocity component if colliding with edge of screen.
        """
        out_left = self.rect.left < screen_rect.left
        out_right = self.rect.right > screen_rect.right
        out_top = self.rect.top < screen_rect.top
        out_bottom = self.rect.bottom > screen_rect.bottom
        if any([out_left, out_right, out_top, out_bottom]):
            self.step_back()
        if out_left or out_right:
            self.vector[0] *= -1
        if out_top or out_bottom:
            self.vector[1] *= -1

        return out_left + out_right + out_top + out_bottom

    def switch_components(self, other, i):
        """Exchange the i component of velocity with other."""
        self.vector[i], other.vector[i] = other.vector[i], self.vector[i]

    def collide(self, others):
        """
        Check collision with other and switch components if hit.
        """
        count = 0
        other = None
        hit = pg.sprite.spritecollideany(self, others, collide_other)
        if hit:
            other = hit
            self.step_back()

        if other:
            if type(other).__name__ not in ["Checkpoint", "Agent"]:
                on_bottom = self.rect.bottom <= other.rect.top
                on_top = self.rect.top >= other.rect.bottom
                self.switch_components(other, on_bottom or on_top)
                # if type(self).__name__ == "Agent" and type(other).__name__ == "Obstacle":
                #     print "I am an agent, I hit an Obstacle."
            else:
                # if type(self).__name__ == "Obstacle" and type(other).__name__ == "Agent":
                #     print "I am an obstacle, I hit an Agent."
                self.vector[0] *= -1
                self.vector[1] *= -1
        return other

    def update(self, screen_rect, others, info):
        """
        Update position; check collision with other blocks; and check
        collision with screen boundaries.
        """
        self.true_pos[0] += self.vector[0]
        self.true_pos[1] += self.vector[1]
        self.rect.center = self.true_pos
        wall_crash = self.collide_walls(screen_rect)
        other_crash = self.collide(others)

        if info is not None:
            assert isinstance(info, dict)

            info.setdefault("wallCrash", 0)
            info.setdefault("obCrash", [])
            info.setdefault("agCrash", [])
            info.setdefault("chCrash", False)
            if type(self).__name__ == "Agent":
                info["wallCrash"] += wall_crash
                if 'Obstacle' in type(other_crash).__name__:
                    info["obCrash"].append(other_crash.identity)
                if 'Agent' in type(other_crash).__name__:
                    info["agCrash"].append(other_crash.identity)
                if 'Checkpoint' in type(other_crash).__name__:
                    info["chCrash"] = True
            elif type(self).__name__ == "Obstacle":
                if 'Agent' in type(other_crash).__name__:
                    info['obCrash'].append(self.identity)
