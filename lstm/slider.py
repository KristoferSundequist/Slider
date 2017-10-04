from graphics import *
import numpy as np
import policy

##########
## GAME ##
##########

width = 500
height = 500
win = GraphWin("canvas", width,height)

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
        Circle(Point(self.x, self.y), self.radius).draw(win)

class Target:
    def __init__(self,radius):
        self.x = np.random.randint(width)
        self.y = np.random.randint(height)
        self.radius = radius

    def reset(self):
        self.x = np.random.randint(width)
        self.y = np.random.randint(height)

    def render(self,win):
        Circle(Point(self.x, self.y), self.radius).draw(win)

def intersect(slider, target):
    return slider.radius + target.radius > np.sqrt(np.power(slider.x - target.x, 2) + np.power(slider.y - target.y, 2))



s = Slider()
t = Target(50)

##############
## RL STUFF ##
##############

state_space_size = 4

def get_state():
    return np.array([s.x/width, s.y/height, t.x/width, t.y/height])

def step(action):
    s.push(action)
    s.update()
    
    reward = 0
    if intersect(s,t):
        reward = 1
        t.reset()

    return reward, get_state()
    
def getEpisode(n):
    policy.agent.initHidden()
    t.reset()
    s.reset()
    
    states = np.zeros((n, state_space_size))
    actionprobs = []
    actions = np.zeros((n,4))
    rewards = np.zeros(n)

    current_state = get_state()
    
    for i in range(n):
        actionprob, action = policy.get_action(current_state)

        states[i] = current_state
        actions[i] = np.eye(4)[action]
        actionprobs.append(actionprob)
        
        reward, current_state = step(action)

        rewards[i] = reward

    return states, actionprobs, actions, rewards

#torch.save(policy.agent.state_dict(), PATH)
#policy.agent.load_state_dict(torch.load(PATH))

running_reward = 0
def train(episode_size, episodes,beta):
    global running_reward
    decay = 0.9
    for i in range(episodes):
        policy.agent.zero_grad()
        states, actionprobs, actions, rewards = getEpisode(episode_size)
        running_reward = decay*running_reward + rewards.sum()*(1-decay)
        print("running: ", running_reward, ", cur: ", rewards.sum())
        policy.train(states, actionprobs, actions, discount(rewards,0.99,0),beta)
        

def discount(r,gamma,init):
    dr = np.zeros_like(r)
    dr[dr.shape[0]-1] = init
    R = init
    for i in reversed(range(len(r)-1)):
        R = r[i] + gamma*R
        dr[i] = R

    dr -= np.mean(dr)
    dr /= (np.std(dr) + 0.0000001)
    return dr

def agent_loop(iterations):
    import time
    #rnn.initHidden()
    for i in range(iterations):
            _,action = policy.get_action(get_state())
            s.push(action)
            s.update()
            if intersect(s,t):
                t.reset()
            
            t.render(win)
            s.render(win)
            time.sleep(0.01)
            clear(win)

def random_loop(iterations):
    import time
    for i in range(iterations):
            s.push(np.random.randint(5))
            s.update()
            if intersect(s,t):
                t.reset()
            
            t.render(win)
            s.render(win)
            time.sleep(0.01)
            clear(win)


