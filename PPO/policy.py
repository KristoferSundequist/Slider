import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
import numpy as np

class policy(nn.Module):
    def __init__(self, state_space_size, action_space_size):
        super(policy, self).__init__()
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

        self.shared = nn.Linear(state_space_size,200)
        self.seq = nn.GRU(200,200)
        
        self.a1 = nn.Linear(200,200)
        self.a2 = nn.Linear(200,200)
        self.action_out = nn.Linear(200, action_space_size)

        self.v1 = nn.Linear(200,200)
        self.v2 = nn.Linear(200,200)
        self.value_out = nn.Linear(200,1)
        
        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            size = m.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            variance = np.sqrt(2.0/(fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)

    def init_hidden(self):
        return torch.randn(1,1,200, requires_grad=False)

    def save_weights(self, name):
        torch.save(self.state_dict(), name)

    def load_weights(self, name):
        self.load_state_dict(torch.load(name))
        
    def forward(self, state, hidden):
        x = F.relu(self.shared(state))
        x, hidden = self.seq(x.view(1,-1,200), hidden)
        x = x.view(-1,200)
        
        a = F.relu(self.a1(x))
        a = F.relu(self.a2(a))
        actions = F.softmax(self.action_out(a), dim=1)

        v = F.relu(self.v1(x))
        v = F.relu(self.v2(v))
        value = self.value_out(v)
        
        return actions, value, hidden

    def get_action(self, state, hidden):
        with torch.no_grad():
            out,value,hidden = self.forward(torch.from_numpy(state).view(1,self.state_space_size).float(), hidden)
            return out, out.multinomial(1).item(), value, hidden
