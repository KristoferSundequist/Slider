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

        self.input_size = 4
        self.hidden_size = 100
        self.output_size = 4
        self.fc1 = nn.Linear(self.input_size,self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, 1)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.hidden_state = None
        self.initHidden()
        
    def forward(self, x, hidden):
        x = F.relu(self.fc1(x))
        x, hidden = self.lstm(x,hidden)
        output = self.out(x.squeeze(1))
        return F.softmax(output), hidden

    def initHidden(self):
        self.hidden_state = (Variable(torch.zeros(1,1, self.hidden_size)), Variable(torch.zeros(1,1, self.hidden_size)))


agent = policy()
agent.apply(weight_init)

opt = optim.RMSprop(agent.parameters(), lr=0.01)

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
    out,hidden = agent(Variable(torch.from_numpy(state)).view(1,1,4).float(), agent.hidden_state)
    agent.hidden_state = hidden
    
    return out, sample(out)
