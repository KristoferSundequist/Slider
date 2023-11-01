import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import copy
import globals
from typing import *
from utils import get_average_gradient


class PolicyNetwork(nn.Module):
    def __init__(self, action_space_size: int, logger):
        super(PolicyNetwork, self).__init__()
        self.logger = logger

        hidden_size = globals.mlp_size
        self.fc1 = nn.Linear(globals.stoch_vector_size + globals.recurrent_vector_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space_size)

        self.apply(self.weight_init)
        self.opt = optim.AdamW(self.parameters(), lr=globals.actor_critic_learning_rate, weight_decay=0.001, eps=globals.actor_critic_adam_eps)

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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.size()[1] == globals.stoch_vector_size + globals.recurrent_vector_size
        x = F.relu(self.fc1(hidden_states))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def update(self, loss: torch.Tensor):
        self.opt.zero_grad()
        loss.backward()
        self.logger.add("Avg policy grad", get_average_gradient(self))
        max_norm_clip = 100
        nn.utils.clip_grad.clip_grad_value_(self.parameters(), max_norm_clip)
        self.opt.step()


def test_policy():
    # Arrange
    hiddens = torch.rand(7, globals.stoch_vector_size + globals.recurrent_vector_size)
    action_space_size = 4
    policyNetwork = PolicyNetwork(action_space_size, None)

    # Act
    logits = policyNetwork.forward(hiddens)

    # Assert
    assert logits.size() == (7, action_space_size)
