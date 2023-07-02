from graphics import *
import numpy as np
import globals

##########
## GAME ##
##########

width = globals.width
height = globals.height

def clear(win):
    for item in win.items[:]:
        item.undraw()
    win.update()


class Slider:

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = np.random.randint(50, width-50)
        self.y = np.random.randint(50, height-50)
        self.radius = 30
        self.speed = 5

    #   1
    # 0   2
    #   3
    def push(self, direction):
        if direction == 0:
            self.x -= self.speed
        elif direction == 1:
            self.y -= self.speed
        elif direction == 2:
            self.x += self.speed
        elif direction == 3:
            self.y += self.speed

    def update(self):

        if self.x + self.radius >= width:
            self.x = width - self.radius

        if self.x - self.radius <= 0:
            self.x = self.radius

        if self.y + self.radius >= height:
            self.y = height - self.radius

        if self.y - self.radius <= 0:
            self.y = self.radius

    def render(self, win):
        c = Circle(Point(self.x, self.y), self.radius)
        c.setFill('black')
        c.draw(win)


class Target:
    def __init__(self, radius):
        self.reset()
        self.radius = radius

    def reset(self):
        self.x = np.random.randint(width)
        self.y = np.random.randint(height)

    def render(self, win):
        c = Circle(Point(self.x, self.y), self.radius)
        c.setFill('yellow')
        c.setOutline('yellow')
        c.draw(win)


class Enemy:
    def __init__(self, radius):
        self.x = np.random.randint(width)
        self.y = np.random.randint(height)
        self.radius = radius

    def reset(self):
        self.x = np.random.randint(width)
        self.y = np.random.randint(height)

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
    state_space_size = 6
    action_space_size = 4

    def __init__(self):
        self.s = Slider()
        self.t = Target(50)
        self.enemy = Enemy(30)

    def intersect(self, a, b):
        return a.radius + b.radius > np.sqrt(np.power(a.x - b.x, 2) + np.power(a.y - b.y, 2))

    def get_state(self):
        # return np.array([s.x/width, s.y/height, s.dx/10, s.dy/10, t.x/width, t.y/height, 0,0])
        return np.array([self.s.x/width, self.s.y/height, self.t.x/width, self.t.y/height, self.enemy.x/width, self.enemy.y/height])

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
    def render(self, left, up, right, down, win):
        clear(win)
        self.t.render(win)
        self.s.render(win)
        self.enemy.render(win)
        Text(Point(250, 300), left).draw(win)
        Text(Point(300, 250), up).draw(win)
        Text(Point(350, 300), right).draw(win)
        Text(Point(300, 350), down).draw(win)
