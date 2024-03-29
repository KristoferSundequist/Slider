from graphics import *
import numpy as np
import policy
import torch


##########
## GAME ##
##########

def clear(win):
    for item in win.items[:]:
        item.undraw()
    win.update()


class Slider:

    def __init__(self, env_width, env_height):
        self.env_width = env_width
        self.env_height = env_height
        self.reset()

    def reset(self):
        self.x = np.random.randint(50, self.env_width - 50)
        self.y = np.random.randint(50, self.env_height - 50)
        self.radius = 30
        self.dx = np.random.randint(-10, 10)
        self.dy = np.random.randint(-10, 10)
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
        else:
            raise Exception(f'Unexpected action: {direction}')

        if self.dx > 10:
            self.dx = 10
        if self.dx < -10:
            self.dx = -10
        if self.dy > 10:
            self.dy = 10
        if self.dy < -10:
            self.dy = -10

    def update(self):

        if self.x + self.radius >= self.env_width:
            self.x = self.env_width - self.radius
            self.dx *= -1

        if self.x - self.radius <= 0:
            self.x = self.radius
            self.dx *= -1

        if self.y + self.radius >= self.env_height:
            self.y = self.env_height - self.radius
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
        c.setFill('black')
        c.draw(win)


class Target:
    def __init__(self, radius, env_width, env_height):
        self.radius = radius
        self.env_width = env_width
        self.env_height = env_height
        self.reset()

    def reset(self):
        self.x = np.random.randint(self.env_width)
        self.y = np.random.randint(self.env_height)

    def render(self, win):
        c = Circle(Point(self.x, self.y), self.radius)
        c.setFill('yellow')
        c.setOutline('yellow')
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
    def __init__(self, radius, env_width, env_height):
        self.x = np.random.randint(env_width)
        self.y = np.random.randint(env_height)
        self.radius = radius
        self.env_width = env_width
        self.env_height = env_height

    def reset(self):
        self.x = np.random.randint(self.env_width)
        self.y = np.random.randint(self.env_height)

    def render(self, win):
        c = Circle(Point(self.x, self.y), self.radius)
        c.setFill('red')
        c.setOutline('red')
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


class Game():
    def __init__(self, env_width, env_height):
        self.s = Slider(env_width, env_height)
        self.t = Target(50, env_width, env_height)
        self.enemy = Enemy(30, env_width, env_height)
        self.env_width = env_width
        self.env_height = env_height

    def intersect(self, a, b):
        return a.radius + b.radius > np.sqrt(np.power(a.x - b.x, 2) + np.power(a.y - b.y, 2))

    def get_state(self):
        return np.array([
            self.s.x / self.env_width,
            self.s.y / self.env_height,
            self.s.dx / 10,
            self.s.dy / 10,
            self.t.x / self.env_width,
            self.t.y / self.env_height,
            self.enemy.x / self.env_width,
            self.enemy.y / self.env_height
        ])

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
    def render(self, win, value):
        clear(win)
        self.t.render(win)
        self.s.render(win)
        self.enemy.render(win)
        Text(Point(250, 300), value).draw(win)
