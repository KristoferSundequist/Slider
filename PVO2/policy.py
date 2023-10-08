import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
import numpy as np
from typing import *


class QNetwork(nn.Module):
    def __init__(self, state_space_size: int, action_space_size: int):
        super(QNetwork, self).__init__()
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

        hidden_size = 512
        self.fc1 = nn.Linear(state_space_size + action_space_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc4 = nn.Linear(hidden_size, 1, bias=False)

        self.opt = optim.Adam(self.parameters(), lr=1e-4, eps=1e-5)

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

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        assert states.size()[1] == self.state_space_size
        assert actions.size()[1] == self.action_space_size
        combined = torch.concat([states, actions], 1)
        assert combined.size()[0] == states.size()[0] == actions.size()[0]
        assert combined.size()[1] == states.size()[1] + actions.size()[1]
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.fc4(x)


def test_q_network():
    qnet = QNetwork(3, 4)
    states = torch.tensor([[0.3, 0, 4], [6, 4, 3]])
    actions = torch.tensor([[0.0, 0, 1, 0], [1, 0, 0, 0]])
    values = qnet.forward(states, actions)
    assert values.size() == (2, 1)
