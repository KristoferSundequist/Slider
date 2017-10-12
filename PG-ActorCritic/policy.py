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

        self.input_size = 6
        self.hidden_size = 100
        self.lstm_size = 64
        self.output_size = 4
        self.fc1 = nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size,self.lstm_size)
        
        self.lstm = nn.LSTM(self.lstm_size, self.lstm_size, 1)
        
        self.out_policy = nn.Linear(self.lstm_size, self.output_size)
        self.out_value = nn.Linear(self.lstm_size, 1)
        
        self.hidden_state = None
        self.initHidden()
        
    def forward(self, x, hidden):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        #x, hidden = self.lstm(x,hidden)
        #output = self.out_policy(x.squeeze(1))
        #value = self.out_value(x.squeeze(1))
        output = self.out_policy(x)
        value = self.out_value(x)
        return F.softmax(output), value, hidden

    def initHidden(self):
        self.hidden_state = (Variable(torch.zeros(1,1, self.lstm_size)), Variable(torch.zeros(1,1, self.lstm_size)))


agent = policy()
agent.apply(weight_init)

opt = optim.Adam(agent.parameters(), lr=0.0003)

def train(states,actionprobs,values,actions,rewards,beta):
    actions = Variable(torch.from_numpy(actions)).float()
    rewards = Variable(torch.from_numpy(rewards)).float()
    values = torch.cat(values).view(len(values)).float()
    actionprobs = torch.cat(actionprobs).float()
    advantage = rewards-values
    #advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

    entropy = -torch.sum(actionprobs * torch.log(actionprobs+1e-10),1)
    policy_loss = -(torch.log(torch.sum(actionprobs*actions,1))*advantage + beta*entropy)
    policy_loss = torch.sum(policy_loss)
    value_loss = torch.sum(0.5*(advantage.pow(2)))

    (policy_loss + 0.5*value_loss).backward()
    nn.utils.clip_grad_norm(agent.parameters(), 40)
    opt.step()

def get_action(state):    
    #out,value,hidden = agent(Variable(torch.from_numpy(state)).view(1,1,4).float(), agent.hidden_state)
    out,value,hidden = agent(Variable(torch.from_numpy(state)).view(1,6).float(), agent.hidden_state)
    agent.hidden_state = hidden

    return out,value,out.multinomial().data[0][0]
