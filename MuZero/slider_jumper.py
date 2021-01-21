import numpy as np
from graphics import *

class Slider:
    
    def __init__(self,env_width, env_height):
        self.x = np.random.randint(50,env_width-50)
        self.y = np.random.randint(50,env_height-50)
        self.radius = 10
        self.dd = 0.5
        self.env_width = env_width
        self.env_height = env_height
        self.speed = 8
        
        self.jump_position = 0.0
        self.jump_velocity = 0.0
        self._gravity = 0.01


    def reset(self):
        self.x = np.random.randint(50,self.env_width-50)
        self.y = np.random.randint(50,self.env_height-50)
        self.dd = 0.5
            
    def _update(self):

        if self.x + self.radius >= self.env_width:
            self.x = self.env_width - self.radius
                    
        if self.x - self.radius <= 0:
            self.x = self.radius
                
        if self.y + self.radius >= self.env_height:
            self.y = self.env_height - self.radius

        if self.y - self.radius <= 0:
            self.y = self.radius
        
        if self.jump_position > 0.0:
            self.jump_position += self.jump_velocity
            self.jump_velocity -= self._gravity
        else:
            self.jump_position = 0.0

    def _jump(self):
        if self.jump_position == 0.0:
            self.jump_velocity = 0.2
            self.jump_position += self.jump_velocity
    
    # 0 : nothing
    #
    #   1
    # 4   2
    #   3
    #
    #5 : jump
    def act(self, action):
        if action == 1:
            self.y -= self.speed
        elif action == 2:
            self.x += self.speed
        elif action == 3:
            self.y += self.speed
        elif action == 4:
            self.x -= self.speed
        elif action == 5:
            self._jump()
        
        self._update()
        
    def render(self,win):
        size = self.radius * (1 + self.jump_position)
        c = Circle(Point(self.x, self.y), size)
        c.setFill('black')
        c.draw(win)

class Target:
    def __init__(self,radius,env_width, env_height):
        self.x = np.random.randint(env_width)
        self.y = np.random.randint(env_height)
        self.radius = radius
        self.env_width = env_width
        self.env_height = env_height

    def reset(self):
        self.x = np.random.randint(self.env_width)
        self.y = np.random.randint(self.env_height)

    def render(self,win):
        c = Circle(Point(self.x, self.y), self.radius)
        c.setFill('yellow')
        c.setOutline('yellow')
        c.draw(win)

class Enemy:
    def __init__(self, radius, speed, env_width, env_height):
        self.x = np.random.randint(env_width)
        self.y = np.random.randint(env_height)
        self.radius = radius
        self.speed = speed
        self.env_width = env_width
        self.env_height = env_height
    

    def reset(self):
        self.x = np.random.randint(self.env_width)
        self.y = np.random.randint(self.env_height)

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

class Game:
    # state_space_size = 6
    state_space_size = 7
    action_space_size = 6
    
    def __init__(self, env_width, env_height):
        self.s = Slider(env_width, env_height)
        self.t = Target(50, env_width, env_height)
        self.enemy = Enemy(30,3,env_width, env_height)
        self.env_width = env_width
        self.env_height = env_height

    def intersect(self, slider: Slider, target):
        if slider.jump_position > 0.0:
            return False
        return slider.radius + target.radius > np.sqrt(np.power(slider.x - target.x, 2) + np.power(slider.y - target.y, 2))

    def get_state(self):
        MAX_JUMP_POSITION = 2.5
        # return np.array([self.s.x/self.env_width, self.s.y/self.env_height, self.t.x/self.env_width, self.t.y/self.env_height, self.enemy.x/self.env_width, self.enemy.y/self.env_height])
        return np.array([
            self.s.x/self.env_width,
            self.s.y/self.env_height,
            self.s.jump_position/MAX_JUMP_POSITION,
            self.t.x/self.env_width,
            self.t.y/self.env_height,
            self.enemy.x/self.env_width,
            self.enemy.y/self.env_height
        ])

    def render(self, win):
        self.t.render(win)
        self.enemy.render(win)
        self.s.render(win)

    def step(self, action):
        self.s.act(action)
        self.enemy.update(self.s.x,self.s.y)

        reward = 0.0
        if self.intersect(self.s, self.t):
            reward += 0.2
            self.t.reset()

        if self.intersect(self.s,self.enemy):
            reward -= 1
            self.enemy.reset()

        return reward, self.get_state()
