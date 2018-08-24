from graphics import *
import numpy as np
import policy
import torch
import time


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
agent = policy.policy()

def get_state():
    #return np.array([s.x/width, s.y/height, s.dx/10, s.dy/10, t.x/width, t.y/height, 0,0])
    return np.array([s.x/width, s.y/height, s.dx/10, s.dy/10, t.x/width, t.y/height, enemy.x/width, enemy.y/height])

def step(action):
    s.push(action)
    s.update()
    enemy.update(s.x,s.y)
    
    reward = 0
    if intersect(s,t):
        reward += .2
        t.reset()
        
    if intersect(s,enemy):
        reward -= 1
        enemy.reset()
        
    return reward, get_state()
    
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


running_reward = None
#train(20,3000,10,0.01,4,0.1,0.99,0.95,0.0001,4000, True)
def train(num_actors,episode_size,episodes,beta,ppo_epochs,eps,gamma,lambd,lr,batch_size, use_critic):
    time_start = time.time()
    global running_reward
    decay = 0.9

    for i in range(episodes):
        states_data = []
        actionprobs_data = []
        actions_data = []
        values_data = []
        advantage_data = []
        returns_data = []

        episode_reward_sums = np.zeros(num_actors)
        for a in range(num_actors):
            states, actionprobs, actions, values, rewards = getEpisode(episode_size)
            episode_reward_sums[a] = rewards.sum()
            states_data.append(states)
            actionprobs_data.append(torch.cat(actionprobs).float().data)
            actions_data.append(actions)
            values = torch.cat(values).view(-1).float().data
            values_data.append(values)
            gaes = generalized_advantage_estimation(rewards,values.numpy(),gamma,lambd,values[-1])
            advantage_data.append(gaes)
            returns = discount(rewards, gamma, values[-1]);
            returns_data.append(returns)

        if running_reward == None:
            running_reward = episode_reward_sums.mean()
        else:
            running_reward = decay*running_reward + (1-decay)*episode_reward_sums.mean()
            
        
        acc_states = np.concatenate(states_data)
        acc_actionprobs = torch.cat(actionprobs_data)
        acc_actions = np.concatenate(actions_data)
        acc_values = torch.cat(values_data)
        acc_returns = np.concatenate(returns_data)
        acc_advantages = np.concatenate(advantage_data)
        a,v,e = agent.train(acc_states, acc_actionprobs, acc_actions, acc_values, acc_returns,acc_advantages,beta,ppo_epochs,eps,lr,batch_size, use_critic)
        #a,v,e = agent.train(acc_states, acc_actionprobs, acc_actions, acc_values, acc_returns,beta,ppo_epochs,eps,lr,batch_size, use_critic)
        print(i/episodes, running_reward, " ", episode_reward_sums.mean(), "Losses (action, value, entropy): ", a, v, e)
    print("Time: ", time.time()-time_start)
    
def generalized_advantage_estimation(rewards,values,gamma,lambd,last):
    rewards = rewards.astype(float)
    values = np.append(values,last)
    gaes = np.zeros_like(rewards)
    gae = 0
    for i in reversed(range(len(gaes))):
        delta = rewards[i] + gamma*values[i+1] - values[i]
        gae = delta + gamma * lambd * gae
        gaes[i] = gae

    return gaes
        
def discount(r,gamma,init):
    dr = np.zeros_like(r)
    dr[dr.shape[0]-1] = init
    R = init
    for i in reversed(range(len(r))):
        R = r[i] + gamma*R
        dr[i] = R

    return dr

def agent_loop(iterations):
    t.reset()
    s.reset()
    enemy.reset()
    import time
    for i in range(iterations):
            _,action,v = agent.get_action(get_state())
            _,_ = step(action)
                
            t.render(win)
            s.render(win)
            enemy.render(win)
            Text(Point(100,100), v.data[0][0]).draw(win)
            time.sleep(0.01)
            clear(win)
