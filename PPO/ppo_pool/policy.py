import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
import numpy as np

class policy(nn.Module):
    def __init__(self):
        super(policy, self).__init__()

        self.ainp = nn.Linear(8,200)
        self.a0 = nn.Linear(200,200)
        self.a1 = nn.Linear(200,200)
        self.action_out = nn.Linear(200,4)

        self.vinp = nn.Linear(8,200)
        self.v0 = nn.Linear(200,200)
        self.v1 = nn.Linear(200,200)
        self.value_out = nn.Linear(200,1)
        
        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            size = m.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            variance = np.sqrt(2.0/(fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)

    def save_weights(self, name):
        torch.save(self.state_dict(), name)

    def load_weights(self, name):
        self.load_state_dict(torch.load(name))
        
    def forward(self, state):
        a = F.relu(self.ainp(state))
        a = F.relu(self.a0(a))
        a = F.relu(self.a1(a))
        actions = F.softmax(self.action_out(a), dim=1)

        v = F.relu(self.vinp(state))        
        v = F.relu(self.v0(v))
        v = F.relu(self.v1(v))
        value = self.value_out(v)
        
        return actions, value

    def get_action(self, state):
        with torch.no_grad():
            out,value = self.forward(torch.from_numpy(state).view(1,8).float())
            return out, out.multinomial(1).item(), value
