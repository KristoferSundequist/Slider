import torch
from torch import autograd, nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def weight_init(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        variance = np.sqrt(2.0/(fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)

class policy(nn.Module):

    def __init__(self):
        super(policy, self).__init__()

        self.fc1 = nn.Linear(6,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,100)
        self.fc4 = nn.Linear(100,4)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x)


agent = policy()
agent.apply(weight_init)

opt = optim.Adam(agent.parameters(), lr=0.0003)

def train(states,old_actionprobs,actions,advantages,beta,ppo_epochs,eps):
    states = torch.from_numpy(states).float()
    actions = torch.from_numpy(actions)
    advantages = torch.from_numpy(advantages).float()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 0.000001)
    actionprobs = old_actionprobs
    
    for _ in range(ppo_epochs):
        sampler = BatchSampler(SubsetRandomSampler(range(len(states))), 100,drop_last=False)
        for indices in sampler:
            agent.zero_grad()
            indices = torch.LongTensor(indices)
            taken_actions = Variable(actions[indices]).float()
            
            new_actionprobs = agent(Variable(states[indices]))
            new_action = torch.sum(new_actionprobs*taken_actions,1)
            
            old_actionprobs = Variable(actionprobs[indices]).float()
            old_action = torch.sum(old_actionprobs*taken_actions,1)
            
            entropy = -torch.sum(new_actionprobs * torch.log(new_actionprobs + 0.000000001),1)
            ratio = new_action/old_action
            
            adv = Variable(advantages[indices])

            surr1 = ratio*adv
            surr2 = torch.clamp(ratio, 1 - eps , 1 + eps)*adv

            loss = -(torch.min(surr1,surr2).mean() + beta*entropy.mean())

            opt.zero_grad()
            loss.backward()
            opt.step()

def get_action(state):    
    out = agent(Variable(torch.from_numpy(state), volatile=True).view(1,6).float())
    
    return out, out.multinomial().data[0][0]