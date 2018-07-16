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

def clear(win):
    for item in win.items[:]:
        item.undraw()
    win.update()

class Slider:
    
    def __init__(self,c,radi):
        self.x = np.random.randint(radi,width-radi)
        self.y = np.random.randint(radi,height-radi)
        self.radius = radi
        self.dx = np.random.randint(-10,10)
        self.dy = np.random.randint(-10,10)
        self.dd = 0.5
        self.color = c
        self.alt = False
        self.mass = radi*radi*3*10


    def reset(self):
        self.x = np.random.randint(self.radius,width-self.radius)
        self.y = np.random.randint(self.radius,height-self.radius)
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

    def updateAsEnemy(self,sliderx,slidery):
        if self.alt:
            self.alt = False
            if self.x > sliderx:
                self.push(0)
            else:
                self.push(2)
        else:
            self.alt = True
            if self.y > slidery:
                self.push(1)
            else:
                self.push(3)
                    
    def render(self,win):
        c = Circle(Point(self.x, self.y), self.radius)
        c.setFill(self.color)
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
        c.setFill('green')
        c.draw(win)

def getDistance(x1,x2,y1,y2):
    return np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))
        
def intersect(slider, target):
    return slider.radius + target.radius > getDistance(slider.x,target.x,slider.y,target.y)



s1 = Slider('yellow',60)
s2 = Slider('red',30)

t = Target(20)

##############
## RL STUFF ##
##############

state_space_size = 10

def get_state():
    return np.array([s1.x/width, s1.y/height, s1.dx/10, s1.dy/10,
                     s2.x/width, s2.y/height, s2.dx/10, s2.dy/10,
                     t.x/width, t.y/height])

def game_reset():
    s1.reset()
    s2.reset()
    t.reset()
    while intersect(s1,s2):
        s2.reset()

def step(action1):
    s1.push(action1)
    s1.update()
    
    s2.updateAsEnemy(t.x,t.y);
    s2.update()

    reward = 0
    
    #if agent collide, do bounce
    if intersect(s1,s2):
        d = getDistance(s1.x, s2.x, s1.y, s2.y)
        nx = (s2.x - s1.x)/d
        ny = (s2.y - s1.y)/d
        p = 2 * (s1.dx * nx + s1.dy*ny - s2.dx * nx - s2.dy*ny) / (s1.mass + s2.mass)

        s1.dx = s1.dx - p*s2.mass*nx
        s1.dy = s1.dy - p*s2.mass*ny
        s2.dx = s2.dx + p*s1.mass*nx
        s2.dy = s2.dy + p*s1.mass*ny
        
        s1.x += 3*s1.dx
        s1.y += 3*s1.dy
        s2.x += 3*s2.dx
        s2.y += 3*s2.dy

    #reward += getDistance(s2.x,t.x,s2.y,t.y)/width

    if intersect(s1,t):
        reward += 1
        t.reset()
    
    if intersect(s2,t):
        reward -= 1
        t.reset()
    
    return reward, get_state()
    
def getEpisode(n):
    game_reset()
    
    states = np.zeros((n, state_space_size))
    
    actionprobs1 = []
    actions1 = np.zeros((n,4))
    values1 = []
    
    rewards1 = np.zeros(n)
    
    current_state = get_state()
    
    for i in range(n):
        actionprob1, action1, value1 = policy.agent1.get_action(current_state)

        states[i] = current_state
        
        actions1[i] = np.eye(4)[action1]
        actionprobs1.append(actionprob1)
        values1.append(value1)
        
        reward, current_state = step(action1)

        rewards1[i] = reward

    return states, actionprobs1, actions1, values1, rewards1

#torch.save(policy.agent.state_dict(), PATH)
#policy.agent.load_state_dict(torch.load(PATH))

#train(10,4000,1000,0.01,5,0.07,0.99,0.95,0.0002,2000)

def train(num_actors,episode_size,episodes,beta,ppo_epochs,eps,gamma,lambd,lr,batch_size):
    running_reward1 = 0
    decay = 0.9

    for i in range(episodes):
        states_data = []
        
        actionprobs_data1 = []
        actions_data1 = []
        values_data1 = []
        advantage_data1 = []

        episode_reward_sums1 = np.zeros(num_actors)
        for a in range(num_actors):
            states, actionprobs1, actions1, values1,rewards1 = getEpisode(episode_size)
            episode_reward_sums1[a] = rewards1.sum()
            states_data.append(states)
            
            actionprobs_data1.append(torch.cat(actionprobs1).float().data)
            actions_data1.append(actions1)
            values1 = torch.cat(values1).view(-1).float().data
            values_data1.append(values1)
            gaes1 = generalized_advantage_estimation(rewards1,values1.numpy(),gamma,lambd,values1[-1])
            #adv = discount(rewards1, gamma, 0)
            advantage_data1.append(gaes1)

        if running_reward1 == 0:
            running_reward1 = episode_reward_sums1.mean()
            
        running_reward1 = decay*running_reward1 + episode_reward_sums1.mean()*(1-decay)
        print("agent1 reward: ", running_reward1, " ", episode_reward_sums1.mean())
        
        acc_states = np.concatenate(states_data)
        
        acc_actionprobs1 = torch.cat(actionprobs_data1)
        acc_actions1 = np.concatenate(actions_data1)
        acc_values1 = torch.cat(values_data1)
        acc_advantage1 = np.concatenate(advantage_data1)

        policy.train(acc_states, acc_actionprobs1, acc_actions1, acc_values1, acc_advantage1,beta,ppo_epochs,eps,lr,batch_size,policy.agent1)
        
    
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
    for i in reversed(range(len(r)-1)):
        R = r[i] + gamma*R
        dr[i] = R

    return dr

def agent_loop(iterations):
    game_reset()
    import time
    r1 = 0
    for i in range(iterations):
            _,action1,v1 = policy.agent1.get_action(get_state())
    
            reward,_ = step(action1)
            r1 += reward
    
            
            t.render(win)
            s1.render(win)
            s2.render(win)
            Text(Point(100,100), r1).draw(win)
            Text(Point(400,500), reward).draw(win)
            Text(Point(500,500), v1).draw(win)
            time.sleep(0.01)
            clear(win)