import torch
from torch import autograd, nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

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

def train(states,actionprobs,actions,rewards,beta):
    actions = Variable(torch.from_numpy(actions)).float()
    rewards = Variable(torch.from_numpy(rewards)).float()
    actionprobs = torch.cat(actionprobs).float()

    entropy = -torch.sum(actionprobs * torch.log(actionprobs+0.0000001),1)
    loss = -(torch.log(torch.sum(actionprobs*actions,1))*rewards + beta*entropy)
    loss = torch.sum(loss)
    loss.backward()
    opt.step()

def sample(d):
    def cate(probs):
        probs /= probs.sum()
        return np.random.choice(np.arange(0,len(probs)), p=probs)
    
    return cate(d.data.numpy()[0])

def get_action(state):    
    out = agent(Variable(torch.from_numpy(state)).view(1,6).float())
    
    return out, sample(out)
