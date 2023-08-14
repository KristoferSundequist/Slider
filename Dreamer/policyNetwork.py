import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import copy
import globals
from typing import *


class PolicyNetwork(nn.Module):
    def __init__(self, action_space_size: int):
        super(PolicyNetwork, self).__init__()

        hidden_size = 256
        self.fc1 = nn.Linear(globals.hidden_vector_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space_size)

        self.apply(self.weight_init)
        self.opt = optim.AdamW(self.parameters(), lr=globals.action_model_learning_rate, weight_decay=0.001)

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

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, List[int]]:
        assert hidden_states.size()[1] == globals.hidden_vector_size
        x = F.relu(self.fc1(hidden_states))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = F.softmax(x, 1)
        sampled_actions_raw = probs.multinomial(1).tolist()
        sampled_actions = [action_list[0] for action_list in sampled_actions_raw]
        return probs, sampled_actions


def test_policy():
    # Arrange
    hiddens = torch.rand(7, globals.hidden_vector_size)
    action_space_size = 4
    policyNetwork = PolicyNetwork(action_space_size)

    # Act
    probs, actions = policyNetwork.forward(hiddens)

    # Assert
    assert probs.size() == (7, action_space_size)
    assert probs.sum(1).mean().item() == 1
    assert len(actions) == (7)
    assert all([action < 4 for action in actions])
