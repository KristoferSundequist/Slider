import numpy as np
from graphics import *

##########
## GAME ##
##########

class Slider:
    
    def __init__(self,width, height):
        self.x = np.random.randint(50,width-50)
        self.y = np.random.randint(50,height-50)
        self.radius = 30
        self.dx = np.random.randint(-10,10)
        self.dy = np.random.randint(-10,10)
        self.dd = 0.5
        self.width = width
        self.height = height


    def reset(self):
        self.x = np.random.randint(50,self.width-50)
        self.y = np.random.randint(50,self.height-50)
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
        self.dx*=0.99
        self.dy*=0.99
    
    #   1
    # 0   2
    #   3
    def move(self, direction):
        speed = 8
        if direction == 0:
            self.x -= speed
        elif direction == 1:
            self.y -= speed
        elif direction == 2:
            self.x += speed
        else:
            self.y += speed
        
        self.dx = 0
        self.dy = 0
        self.update()
        
                    
    def render(self,win):
        c = Circle(Point(self.x, self.y), self.radius)
        c.setFill('black')
        c.draw(win)

class Target:
    def __init__(self,radius,width, height):
        self.x = np.random.randint(width)
        self.y = np.random.randint(height)
        self.radius = radius
        self.width = width
        self.height = height

    def reset(self):
        self.x = np.random.randint(self.width)
        self.y = np.random.randint(self.height)

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
    def __init__(self,radius,speed,width, height):
        self.x = np.random.randint(width)
        self.y = np.random.randint(height)
        self.radius = radius
        self.speed = speed
        self.width = width
        self.height = height
    

    def reset(self):
        self.x = np.random.randint(self.width)
        self.y = np.random.randint(self.height)

    def render(self,win):
        c = Circle(Point(self.x, self.y), self.radius)
        c.setFill('red')
        c.setOutline('red')
        c.draw(win)

    def update(self,sliderx,slidery):
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
    # state_space_size = 6
    state_space_size = 8
    action_space_size = 4
    
    def __init__(self, width, height):
        self.s = Slider(width, height)
        self.t = Target(50, width, height)
        self.enemy = Enemy(30,3,width, height)
        self.width = width
        self.height = height

    def intersect(self, slider, target):
        return slider.radius + target.radius > np.sqrt(np.power(slider.x - target.x, 2) + np.power(slider.y - target.y, 2))

    def get_state(self):
        # return np.array([self.s.x/self.width, self.s.y/self.height, self.t.x/self.width, self.t.y/self.height, self.enemy.x/self.width, self.enemy.y/self.height])
        return np.array([
            self.s.x/self.width,
            self.s.y/self.height,
            self.s.dx/10,
            self.s.dy/10,
            self.t.x/self.width,
            self.t.y/self.height,
            self.enemy.x/self.width,
            self.enemy.y/self.height
        ])

    def render(self, win):
        self.t.render(win)
        self.s.render(win)
        self.enemy.render(win)

    def step(self, action):
        self.s.push(action)
        self.s.update()
        self.enemy.update(self.s.x,self.s.y)

        reward = 0
        if self.intersect(self.s, self.t):
            reward += 1
            self.t.reset()

        if self.intersect(self.s,self.enemy):
            reward -= 1
            self.enemy.reset()

        return reward, self.get_state()

##########################################
## GAME 2 ################################
##########################################

class Game2:
    # state_space_size = 6
    state_space_size = 10
    action_space_size = 4
    
    def __init__(self, width, height):
        self.s = Slider(width, height)
        self.t = Target(50, width, height)
        self.enemy = Enemy(30,1,width, height)
        self.enemy2 = Enemy(20,3,width, height)
        self.width = width
        self.height = height

    def intersect(self, slider, target):
        return slider.radius + target.radius > np.sqrt(np.power(slider.x - target.x, 2) + np.power(slider.y - target.y, 2))

    def get_state(self):
        # return np.array([self.s.x/self.width, self.s.y/self.height, self.t.x/self.width, self.t.y/self.height, self.enemy.x/self.width, self.enemy.y/self.height])
        return np.array([
            self.s.x/self.width,
            self.s.y/self.height,
            self.s.dx/10,
            self.s.dy/10,
            self.t.x/self.width,
            self.t.y/self.height,
            self.enemy.x/self.width,
            self.enemy.y/self.height,
            self.enemy2.x/self.width,
            self.enemy2.y/self.height
        ])

    def render(self, win):
        self.t.render(win)
        self.s.render(win)
        self.enemy.render(win)
        self.enemy2.render(win)

    def step(self, action):
        self.s.push(action)
        self.s.update()
        self.enemy.update(self.s.x,self.s.y)
        self.enemy2.update(self.s.x,self.s.y)

        reward = 0
        if self.intersect(self.s, self.t):
            reward += .2
            self.t.reset()

        if self.intersect(self.s,self.enemy):
            reward -= 1
            self.enemy.reset()

        if self.intersect(self.s,self.enemy2):
            reward -= 1
            self.enemy2.reset()

        return reward, self.get_state()

##########################################
## GAME 3 - SIMPLE ################################
##########################################

class GameSimple:
    state_space_size = 4
    action_space_size = 4
    
    def __init__(self, width, height):
        self.s = Slider(width, height)
        self.t = Target(50, width, height)
        self.width = width
        self.height = height

    def intersect(self, slider, target):
        return slider.radius + target.radius > np.sqrt(np.power(slider.x - target.x, 2) + np.power(slider.y - target.y, 2))

    def get_state(self):
        return np.array([
            self.s.x/self.width,
            self.s.y/self.height,
            self.t.x/self.width,
            self.t.y/self.height,
        ])

    def render(self, win):
        self.t.render(win)
        self.s.render(win)

    def step(self, action):
        self.s.move(action)

        reward = 0
        if self.intersect(self.s, self.t):
            reward += 1
            self.t.reset()

        return reward, self.get_state()



##########################################
## GAME 4 - BE NEAR ################################
##########################################

class GameBeNear:
    state_space_size = 4
    action_space_size = 4
    
    def __init__(self, width, height):
        self.s = Slider(width, height)
        self.t = Target(50, width, height)
        self.width = width
        self.height = height


    def distance(self, slider, target):
        return np.sqrt(np.power(slider.x - target.x, 2) + np.power(slider.y - target.y, 2))

    def intersect(self, slider, target):
        return slider.radius + target.radius > distance(slider, target)

    def get_state(self):
        return np.array([
            self.s.x/self.width,
            self.s.y/self.height,
            self.t.x/self.width,
            self.t.y/self.height,
        ])

    def render(self, win):
        self.t.render(win)
        self.s.render(win)

    def step(self, action):
        self.s.move(action)

        max_reward = 2
        distance = self.distance(self.s, self.t)
        reward = max_reward if distance == 0 else np.clip(10/distance, -max_reward, max_reward)
        if(np.random.randint(3000) == 0):
            self.t.reset()

        return reward, self.get_state()
