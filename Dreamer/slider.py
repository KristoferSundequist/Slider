from graphics import *
import numpy as np
import globals
from typing import List


##########
## GAME ##
##########


def clear(win):
    for item in win.items[:]:
        item.undraw()
    win.update()


class Slider:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = float(np.random.randint(50, globals.width - 50))
        self.y = float(np.random.randint(50, globals.height - 50))
        self.radius = 30
        self.dx = float(np.random.randint(-10, 10))
        self.dy = float(np.random.randint(-10, 10))
        self.dd = 0.5

    #   1
    # 0   2
    #   3
    def push(self, direction):
        if direction == 0:
            self.dx -= self.dd
        elif direction == 1:
            self.dy -= self.dd
        elif direction == 2:
            self.dx += self.dd
        elif direction == 3:
            self.dy += self.dd

        if self.dx > 10:
            self.dx = 10
        if self.dx < -10:
            self.dx = -10
        if self.dy > 10:
            self.dy = 10
        if self.dy < -10:
            self.dy = -10

    def update(self):
        if self.x + self.radius >= globals.width:
            self.x = globals.width - self.radius
            self.dx *= -1

        if self.x - self.radius <= 0:
            self.x = self.radius
            self.dx *= -1

        if self.y + self.radius >= globals.height:
            self.y = globals.height - self.radius
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
    def __init__(self, radius):
        self.reset()
        self.radius = radius

    def reset(self):
        self.x = float(np.random.randint(globals.width))
        self.y = float(np.random.randint(globals.height))

    def render(self, win):
        c = Circle(Point(self.x, self.y), self.radius)
        c.setFill("yellow")
        c.setOutline("yellow")
        c.draw(win)


class Enemy:
    def __init__(self, radius, speed):
        self.x = float(np.random.randint(globals.width))
        self.y = float(np.random.randint(globals.height))
        self.radius = radius
        self.speed = speed

    def reset(self):
        self.x = float(np.random.randint(globals.width))
        self.y = float(np.random.randint(globals.height))

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


class Game:
    state_space_size = 8
    action_space_size = 5

    def __init__(self):
        self.s = Slider()
        self.t = Target(50)
        self.enemy = Enemy(30, 1)

    def intersect(self, a, b):
        return a.radius + b.radius > np.sqrt(np.power(a.x - b.x, 2) + np.power(a.y - b.y, 2))

    def get_state(self) -> List[float]:
        return [
            self.s.x / globals.width,
            self.s.y / globals.height,
            self.s.dx / 10,
            self.s.dy / 10,
            self.t.x / globals.width,
            self.t.y / globals.height,
            self.enemy.x / globals.width,
            self.enemy.y / globals.height,
        ]

    def set_game_state(self, gamestate: List[float]):
        self.s.x = gamestate[0] * globals.width
        self.s.y = gamestate[1] * globals.height
        self.s.dx = gamestate[2] * 10
        self.s.dy = gamestate[3] * 10
        self.t.x = gamestate[4] * globals.width
        self.t.y = gamestate[5] * globals.height
        self.enemy.x = gamestate[6] * globals.width
        self.enemy.y = gamestate[7] * globals.height
        

    def step(self, action):
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

    #   1
    # 0   2
    #   3
    def render(self, value, reward, win):
        clear(win)
        self.t.render(win)
        self.s.render(win)
        self.enemy.render(win)
        Text(Point(250, 250), value).draw(win)
        Text(Point(300, 300), reward).draw(win)

class GameWithoutSpeed:
    state_space_size = 6
    action_space_size = 5

    def __init__(self):
        self.s = Slider()
        self.t = Target(50)
        self.enemy = Enemy(30, 1)

    def intersect(self, a, b):
        return a.radius + b.radius > np.sqrt(np.power(a.x - b.x, 2) + np.power(a.y - b.y, 2))

    def get_state(self) -> List[float]:
        return [
            self.s.x / globals.width,
            self.s.y / globals.height,
            self.t.x / globals.width,
            self.t.y / globals.height,
            self.enemy.x / globals.width,
            self.enemy.y / globals.height,
        ]

    def set_game_state(self, gamestate: List[float]):
        self.s.x = gamestate[0] * globals.width
        self.s.y = gamestate[1] * globals.height
        self.t.x = gamestate[2] * globals.width
        self.t.y = gamestate[3] * globals.height
        self.enemy.x = gamestate[4] * globals.width
        self.enemy.y = gamestate[5] * globals.height
        

    def step(self, action):
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

    #   1
    # 0   2
    #   3
    def render(self, value, reward, win):
        clear(win)
        self.t.render(win)
        self.s.render(win)
        self.enemy.render(win)
        Text(Point(250, 250), value).draw(win)
        Text(Point(300, 300), reward).draw(win)
