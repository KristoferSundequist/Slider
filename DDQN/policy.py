import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import copy
    

class policy(nn.Module):
    def __init__(self):
        super(policy, self).__init__()
        self.n_inputs = 8
        self.n_outputs = 4
        self.hidden_size = 256
        
        self.fc1 = nn.Linear(self.n_inputs, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.action_out = nn.Linear(self.hidden_size, self.n_outputs)
        
        #self.value_features = nn.Linear(200,100)
        #self.value = nn.Linear(100,1)

        #self.advantage_features = nn.Linear(200,100)
        #self.advantages = nn.Linear(100,self.n_outputs)
        
        self.apply(self.weight_init)
        self.opt = optim.Adam(self.parameters(), lr=1e-4)

    def copy_weights(self, other):
        self.load_state_dict(copy.deepcopy(other.state_dict()))

    def save_weights(self, name):
        torch.save(self.state_dict(), name)

    def load_weights(self, name):
        self.load_state_dict(torch.load(name))
        
    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            size = m.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            variance = np.sqrt(2.0/(fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        qvals = self.action_out(x)
        return qvals

        #vf = F.relu(self.value_features(x))
        #v = self.value(vf)

        #af = F.relu(self.advantage_features(x))
        #a = self.advantages(af)

        #return v + a - a.mean()

    def get_action(self, state):
        with torch.no_grad():
            q = self.forward(torch.from_numpy(state).view(1,8).float())
            return torch.max(q,1)[1].data[0]

