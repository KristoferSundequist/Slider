import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import copy
import globals
from typing import *


class ReconstructionNetwork(nn.Module):
    def __init__(self, state_space_size):
        super(ReconstructionNetwork, self).__init__()
        self.state_space_size = state_space_size

        hidden_size = 256
        self.fc1 = nn.Linear(globals.hidden_vector_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, state_space_size)

        self.apply(self.weight_init)
        self.opt = optim.AdamW(self.parameters(), lr=globals.world_model_learning_rate, weight_decay=0.001)

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
            variance = np.sqrt(2.0 / (fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)

    def forward(self, hiddens: torch.Tensor) -> torch.Tensor:
        assert hiddens.size()[1] == globals.hidden_vector_size
        x = F.relu(self.fc1(hiddens))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def test_reconstuction():
    # Arrange
    hiddens = torch.rand(2, globals.hidden_vector_size)
    reconstruction = ReconstructionNetwork(3)

    # Act
    result = reconstruction.forward(hiddens)

    # Assert
    assert result.size() == (2, 3)
