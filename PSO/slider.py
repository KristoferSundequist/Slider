from graphics import *
import numpy as np


##########
## GAME ##
##########

width = 700
height = 700
win = GraphWin("canvas", width,height)
win.setBackground('lightskyblue')
    
def clear(win):
    for item in win.items[:]:
        item.undraw()
    win.update()

class Slider:
    
    def __init__(self):
        self.x = np.random.randint(50,width-50)
        self.y = np.random.randint(50,height-50)
        self.radius = 30
        self.dx = np.random.randint(-10,10)
        self.dy = np.random.randint(-10,10)
        self.dd = 0.5


    def reset(self):
        self.x = np.random.randint(50,width-50)
        self.y = np.random.randint(50,height-50)
        self.radius = 30
        self.dx = np.random.randint(-10,10)
        self.dy = np.random.randint(-10,10)
        self.dd = 0.5
    
    #   1
    # 0   2
    #   3
    def push(self,direction):
        if direction == 0:
            self.dx -= self.dd
        elif direction == 1:
            self.dy -= self.dd
        elif direction == 2:
            self.dx += self.dd
        elif direction == 3:
            self.dy += self.dd

        if self.dx > 10: self.dx = 10
        if self.dx < -10: self.dx = -10
        if self.dy > 10: self.dy = 10
        if self.dy < -10: self.dy = -10
            
    def update(self):

        if self.x + self.radius >= width:
            self.x = width - self.radius
            self.dx *= -1
                    
        if self.x - self.radius <= 0:
            self.x = self.radius
            self.dx *= -1
                
        if self.y + self.radius >= height:
            self.y = height - self.radius
            self.dy *= -1

        if self.y - self.radius <= 0:
            self.y = self.radius
            self.dy *= -1
                
        self.x += self.dx
        self.y += self.dy
        self.dx*=0.99
        self.dy*=0.99
        
                    
    def render(self,win):
        c = Circle(Point(self.x, self.y), self.radius)
        c.setFill('black')
        c.draw(win)

class Target:
    def __init__(self,radius):
        self.x = np.random.randint(width)
        self.y = np.random.randint(height)
        self.radius = radius

    def reset(self):
        self.x = np.random.randint(width)
        self.y = np.random.randint(height)

    def render(self,win):
        c = Circle(Point(self.x, self.y), self.radius)
        c.setFill('yellow')
        c.setOutline('yellow')
        c.draw(win)

    def update(self,sliderx,slidery):
        if self.x > sliderx:
            self.x -= 1
        else:
            self.x += 1

        if self.y > slidery:
            self.y -= 1
        else:
            self.y += 1

class Enemy:
    def __init__(self,radius):
        self.x = np.random.randint(width)
        self.y = np.random.randint(height)
        self.radius = radius

    def reset(self):
        self.x = np.random.randint(width)
        self.y = np.random.randint(height)

    def render(self,win):
        c = Circle(Point(self.x, self.y), self.radius)
        c.setFill('red')
        c.setOutline('red')
        c.draw(win)

    def update(self,sliderx,slidery):
        if self.x > sliderx:
            self.x -= 1
        else:
            self.x += 1

        if self.y > slidery:
            self.y -= 1
        else:
            self.y += 1
               
def intersect(slider, target):
    return slider.radius + target.radius > np.sqrt(np.power(slider.x - target.x, 2) + np.power(slider.y - target.y, 2))

s = Slider()
t = Target(50)
enemy = Enemy(30)

##############
## RL STUFF ##
##############

state_space_size = 8

def get_state():
    #return np.array([s.x/width, s.y/height, s.dx/10, s.dy/10, t.x/width, t.y/height, 0, 0])
    return np.array([s.x/width, s.y/height, s.dx/10, s.dy/10, t.x/width, t.y/height, enemy.x/width, enemy.y/height])

def step(action):
    s.push(action)
    s.update()
    enemy.update(s.x,s.y)
    
    reward = 0
    if intersect(s,t):
        reward += 1
        t.reset()
        
    if intersect(s,enemy):
        reward -= 5
        enemy.reset()
        
    return reward, get_state()    

def reset():
    s.reset()
    t.reset()
    enemy.reset()

def render():
    clear(win)
    t.render(win)
    s.render(win)
    enemy.render(win)
    
