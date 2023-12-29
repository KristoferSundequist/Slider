import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
import numpy as np
from typing import *


class TheNetwork(nn.Module):
    def __init__(self, state_space_size, action_space_size):
        super(TheNetwork, self).__init__()
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

        value_hidden_size = 510
        self.value_1 = nn.Linear(state_space_size, value_hidden_size, bias=False)
        self.value_2 = nn.Linear(value_hidden_size, value_hidden_size, bias=False)
        self.value_out = nn.Linear(value_hidden_size, 255, bias=False)

        action_hidden_size = 255
        self.action_1 = nn.Linear(state_space_size, action_hidden_size, bias=False)
        self.action_2 = nn.Linear(action_hidden_size, action_hidden_size, bias=False)
        self.action_out = nn.Linear(action_hidden_size, action_space_size * 2, bias=False)

        self.opt = optim.Adam(self.parameters(), lr=3e-4, eps=1e-5)

        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            size = m.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            variance = np.sqrt(2.0 / (fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)

    def init_hidden(self):
        return torch.randn(1, 1, 200, requires_grad=False)

    def save_weights(self, name):
        torch.save(self.state_dict(), name)

    def load_weights(self, name):
        self.load_state_dict(torch.load(name, map_location="cpu"))

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert states.size()[1] == self.state_space_size

        value = F.relu(self.value_1(states))
        value = F.relu(self.value_2(value))
        value = self.value_out(value)

        action = F.relu(self.action_1(states))
        action = F.relu(self.action_2(action))
        action = self.action_out(action)
        [action_mean, action_std] = torch.chunk(action, 2, 1)

        return value, F.tanh(action_mean), F.tanh(action_std)


def test_network():
    net = TheNetwork(3, 4)
    states = torch.tensor([[0.3, 0, 4], [6, 4, 3]])
    values, action_mean, action_std = net.forward(states)
    assert values.size() == (2, 1)
    assert action_mean.size() == (2, 4)
    assert action_std.size() == (2, 4)
