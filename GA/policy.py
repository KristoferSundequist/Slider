import torch
from torch import autograd, nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import time
from random import randint
import copy
import slider

class policy(nn.Module):
    def __init__(self,sigma):
        super(policy, self).__init__()

        self.sigma = sigma
        self.fitness_score = 0
        
        self.fc1 = nn.Linear(8,200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,200)
        self.action_out = nn.Linear(200,4)
        
        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            size = m.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            variance = np.sqrt(2.0/(fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)

    def mutate_layer(self,layer):
        if isinstance(layer, nn.Linear):
            dist = [0.9,0.1]
            maskw = np.random.choice(2,(layer.weight.shape[0], layer.weight.shape[1]),p=dist)
            maskb = np.random.choice(2,layer.bias.shape[0],p=dist)
            
            layer.weight.data += self.sigma*torch.randn(layer.weight.shape)*torch.from_numpy(maskw).float()
            layer.bias.data += self.sigma*torch.randn(layer.bias.shape)*torch.from_numpy(maskb).float()

    def mutate(self):
        self.apply(self.mutate_layer)
        
    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        
        actions = self.action_out(x)
        #return actions
        return F.softmax(actions,dim=1)

    def get_action(self, state):    
        out = self.forward(Variable(torch.from_numpy(state), volatile=True).view(1,8).float())
        #return torch.max(out,1)[1].data[0]
        return out.multinomial().data[0][0]

    def fitness2(self,iters):
        a = self.get_action(np.array([   1, 0.2,   1, 0.3,  1, 0.1, 1,   1]))
        b = self.get_action(np.array([   1,   0, 0.1,  0.6,  1,   1, 0,  1]))
        c = self.get_action(np.array([   0,   1,   0,   1,  0,   1, 0,   -1]))
        d = self.get_action(np.array([   0,   1,   1,   1, -1,   1, 1,   -1]))
        e = self.get_action(np.array([-0.1,   0,  -1,   1,0.5, 0.6, 1,   1]))
        f = self.get_action(np.array([   0, 0.1,-0.5, 0.7,  1,   1, 1, 0.4]))
        g = self.get_action(np.array([-0.3,   0,-0.3, 0.2,0.8,   1, 1,   1]))
        h = self.get_action(np.array([ 0.8, 0.5, 0.5, 0.2,  1,   1, 0,   1]))
        i = self.get_action(np.array([ 0.5, 0.9,  -1,-0.5,  1,-0.5, 1,   0]))
        j = self.get_action(np.array([   1,   1,  -1,-0.2,0.8,-0.8, 1,-0.3]))
        
        fit = int(a==0) + int(b==0) + int(c==1) + int(d==1) + int(e==2) + int(f==2) + int(g==2) + int(h==0) + int(i==3) + int(j==3)
        self.fitness_score = fit

    # Eval some weights for iters steps (episode size)
    def fitness(self,iters):
        slider.reset()
        accReward = 0
        state = slider.get_state()
        for _ in range(iters):
            action = self.get_action(state)
            reward,state = slider.step(action)
            accReward += reward
            
        self.fitness_score = accReward
        #return accReward# - l2loss(b1,w1,b2,w2,b3,w3)

    # Eval current weights in env (wrapper)
    def fitness_current(self,iters):
        return fitness(iters)

    # Watch agent do its thing
    def agent_loop(self,iters):
        slider.reset()
        for _ in range(iters):
            state = slider.get_state()
            action = self.get_action(state)
            _,_ = slider.step(action)
            slider.render()
            time.sleep(0.01)


class Population:
    def __init__(self,pop_size,sigma,iters):
        self.iters = iters
        self.pop_size = pop_size
        self.sigma = sigma
        self.Nns = [policy(sigma) for _ in range(pop_size)]
        for i in range(pop_size):
            self.Nns[i].fitness(self.iters)
        self.Nns.sort(key = lambda x: -x.fitness_score)

    # precondition: Nns sorted by decreasing fitness
    def next_generation(self, pop_size, T, iters):
        new_pop = []
        new_pop.append(self.Nns[0])
        
        parents = [randint(0,T-1) for _ in range(pop_size-1)]
        for p in parents:
            child = copy.deepcopy(self.Nns[p])
            child.mutate()
            new_pop.append(child)

        for nn in new_pop:
            nn.fitness(iters)
            
        self.Nns = new_pop
        self.Nns.sort(key = lambda x: -x.fitness_score)

    def nGens(self,n,pop_size,T,iters):
        start = time.time()
        for i in range(n):
            self.next_generation(pop_size,T,iters)
            print(i, self.Nns[0].fitness_score)
        end = time.time()
        print(end-start)
            
pop = Population(1000,0.01,3000)