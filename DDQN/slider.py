from graphics import *
import numpy as np
import policy
import torch


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
        self.reset()

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
        self.reset()
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

class Game():
    def __init__(self):
        self.s = Slider()
        self.t = Target(50)
        self.enemy = Enemy(30)

    def intersect(self,a,b):
        return a.radius + b.radius > np.sqrt(np.power(a.x - b.x, 2) + np.power(a.y - b.y, 2))
    
    def get_state(self):
        #return np.array([s.x/width, s.y/height, s.dx/10, s.dy/10, t.x/width, t.y/height, 0,0])
        return np.array([self.s.x/width, self.s.y/height, self.s.dx/10, self.s.dy/10, self.t.x/width, self.t.y/height, self.enemy.x/width, self.enemy.y/height])
    
    def step(self, action):
        self.s.push(action)
        self.s.update()
        self.enemy.update(self.s.x,self.s.y)
        
        reward = 0
        if self.intersect(self.s,self.t):
            reward += 1
            self.t.reset()
            
        if self.intersect(self.s,self.enemy):
            reward -= 5
            self.enemy.reset()
        
        return reward, self.get_state()

    #   1
    # 0   2
    #   3
    def render(self, left, up, right, down):
        clear(win)
        self.t.render(win)
        self.s.render(win)
        self.enemy.render(win)
        Text(Point(250,300), left).draw(win)
        Text(Point(300,250), up).draw(win)
        Text(Point(350,300), right).draw(win)
        Text(Point(300,350), down).draw(win)
        




##############
## RL STUFF ##
##############

#state_space_size = 8
#agent = policy.policy()

def getEpisode(n):
    t.reset()
    s.reset()
    enemy.reset()
    
    states = np.zeros((n, state_space_size))
    actionprobs = []
    actions = np.zeros((n,4))
    values = []
    rewards = np.zeros(n)
    
    current_state = get_state()
    
    for i in range(n):
        actionprob, action, value = agent.get_action(current_state)

        states[i] = current_state
        actions[i] = np.eye(4)[action]
        actionprobs.append(actionprob)
        values.append(value)
        
        reward, current_state = step(action)

        rewards[i] = reward

    return states, actionprobs, actions, values, rewards

#torch.save(policy.agent.state_dict(), PATH)
#policy.agent.load_state_dict(torch.load(PATH))


    


