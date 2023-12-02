import numpy as np
from graphics import *
from typing import *

##########
## GAME ##
##########


class Slider:
    def __init__(self, width, height):
        self.x = np.random.randint(50, width - 50)
        self.y = np.random.randint(50, height - 50)
        self.radius = 30
        self.dx = np.random.randint(-10, 10)
        self.dy = np.random.randint(-10, 10)
        self.dd = 0.5
        self.width = width
        self.height = height

    def reset(self):
        self.x = np.random.randint(50, self.width - 50)
        self.y = np.random.randint(50, self.height - 50)
        self.radius = 30
        self.dx = np.random.randint(-10, 10)
        self.dy = np.random.randint(-10, 10)
        self.dd = 0.5

    #   1
    # 0   2
    #   3
    def push(self, direction: List[float]):
        direction_normalizer = max(sum(direction), 1.0)
        self.dx += direction[0] / direction_normalizer
        self.dy += direction[1] / direction_normalizer

        if self.dx > 10:
            self.dx = 10
        if self.dx < -10:
            self.dx = -10
        if self.dy > 10:
            self.dy = 10
        if self.dy < -10:
            self.dy = -10

    def update(self):
        if self.x + self.radius >= self.width:
            self.x = self.width - self.radius
            self.dx *= -1

        if self.x - self.radius <= 0:
            self.x = self.radius
            self.dx *= -1

        if self.y + self.radius >= self.height:
            self.y = self.height - self.radius
            self.dy *= -1

        if self.y - self.radius <= 0:
            self.y = self.radius
            self.dy *= -1

        self.x += self.dx
        self.y += self.dy
        self.dx *= 0.99
        self.dy *= 0.99

    def render(self, win):
        c = Circle(Point(self.x, self.y), self.radius)
        c.setFill("black")
        c.draw(win)


class Target:
    def __init__(self, radius, width, height):
        self.x = np.random.randint(width)
        self.y = np.random.randint(height)
        self.radius = radius
        self.width = width
        self.height = height

    def reset(self):
        self.x = np.random.randint(self.width)
        self.y = np.random.randint(self.height)

    def render(self, win):
        c = Circle(Point(self.x, self.y), self.radius)
        c.setFill("yellow")
        c.setOutline("yellow")
        c.draw(win)

    def update(self, sliderx, slidery):
        if self.x > sliderx:
            self.x -= 1
        else:
            self.x += 1

        if self.y > slidery:
            self.y -= 1
        else:
            self.y += 1


class Enemy:
    def __init__(self, radius, speed, width, height):
        self.x = np.random.randint(width)
        self.y = np.random.randint(height)
        self.radius = radius
        self.speed = speed
        self.width = width
        self.height = height

    def reset(self):
        self.x = np.random.randint(self.width)
        self.y = np.random.randint(self.height)

    def render(self, win):
        c = Circle(Point(self.x, self.y), self.radius)
        c.setFill("red")
        c.setOutline("red")
        c.draw(win)

    def update(self, sliderx, slidery):
        if self.x > sliderx:
            self.x -= self.speed
        else:
            self.x += self.speed

        if self.y > slidery:
            self.y -= self.speed
        else:
            self.y += self.speed


##########################################
## GAME 1 ################################
##########################################


class Game:
    state_space_size = 8
    action_space_size = 2

    def __init__(self, width, height):
        self.s = Slider(width, height)
        self.t = Target(50, width, height)
        self.enemy = Enemy(30, 1, width, height)
        self.width = width
        self.height = height

    def intersect(self, slider, target):
        return slider.radius + target.radius > np.sqrt(
            np.power(slider.x - target.x, 2) + np.power(slider.y - target.y, 2)
        )

    def get_state(self) -> List[float]:
        return [
            self.s.x / self.width,
            self.s.y / self.height,
            self.s.dx / 10,
            self.s.dy / 10,
            self.t.x / self.width,
            self.t.y / self.height,
            self.enemy.x / self.width,
            self.enemy.y / self.height,
        ]

    def render(self, win):
        self.t.render(win)
        self.s.render(win)
        self.enemy.render(win)

    def step(self, action: List[float]):
        self.s.push(action)
        self.s.update()
        self.enemy.update(self.s.x, self.s.y)

        reward = 0
        if self.intersect(self.s, self.t):
            reward += 0.2
            self.t.reset()

        if self.intersect(self.s, self.enemy):
            reward -= 1
            self.enemy.reset()

        return reward, self.get_state()


##########################################
## GAME 2 ( 2 sliders ) #########
##########################################


class GameTwo:
    state_space_size = 12
    action_space_size = 4

    def __init__(self, width, height):
        self.s1 = Slider(width, height)
        self.s2 = Slider(width, height)
        self.t = Target(50, width, height)
        self.enemy = Enemy(30, 1, width, height)
        self.width = width
        self.height = height

    def get_distance(self, a, b):
        return np.sqrt(np.power(a.x - b.x, 2) + np.power(a.y - b.y, 2)) - (a.radius + b.radius)

    def intersect(self, slider, target):
        return self.get_distance(slider, target) <= 0

    def get_state(self) -> List[float]:
        return [
            self.s1.x / self.width,
            self.s1.y / self.height,
            self.s1.dx / 10,
            self.s1.dy / 10,
            self.s2.x / self.width,
            self.s2.y / self.height,
            self.s2.dx / 10,
            self.s2.dy / 10,
            self.t.x / self.width,
            self.t.y / self.height,
            self.enemy.x / self.width,
            self.enemy.y / self.height,
        ]

    def render(self, win):
        self.t.render(win)
        self.s1.render(win)
        self.s2.render(win)
        self.enemy.render(win)

    def step(self, action: List[float]):
        self.s1.push(action[:2])
        self.s1.update()
        self.s2.push(action[2:])
        self.s2.update()

        if self.get_distance(self.s1, self.t) < self.get_distance(self.s2, self.t):
            self.enemy.update(self.s1.x, self.s1.y)
        else:
            self.enemy.update(self.s2.x, self.s2.y)

        reward = 0
        if self.intersect(self.s1, self.t) or self.intersect(self.s2, self.t):
            reward += 0.2
            self.t.reset()

        if self.intersect(self.s1, self.enemy) or self.intersect(self.s2, self.enemy):
            reward -= 1
            self.enemy.reset()

        return reward, self.get_state()
