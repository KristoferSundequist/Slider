import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import copy
import globals
from typing import *


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()

        hidden_size = 128
        self.fc1 = nn.Linear(globals.hidden_vector_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        self.apply(self.weight_init)
        self.opt = optim.AdamW(self.parameters(), lr=globals.value_model_learning_rate, weight_decay=0.001)

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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.size()[1] == globals.hidden_vector_size
        x = F.relu(self.fc1(hidden_states))
        return self.fc2(x)

def test_value():
    # Arrange
    hiddens = torch.rand(7, globals.hidden_vector_size)
    valueNetwork = ValueNetwork()

    # Act
    result = valueNetwork.forward(hiddens)

    # Assert
    assert result.size() == (7, 1)
