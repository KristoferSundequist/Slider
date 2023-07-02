import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import copy
import globals
from typing import *


class Policy(nn.Module):
    def __init__(self, state_space_size, action_space_size):
        super(Policy, self).__init__()
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.hidden_size = 512

        self.fc1 = nn.Linear(state_space_size, self.hidden_size)
        # self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.action_out = nn.Linear(self.hidden_size, self.n_outputs)

        #self.value_features = nn.Linear(state_space_size, self.hidden_size)
        self.value_features2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, 1)

        #self.advantage_features = nn.Linear(state_space_size, self.hidden_size)
        self.advantage_features2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.advantages = nn.Linear(self.hidden_size, action_space_size)

        self.apply(self.weight_init)
        self.opt = optim.Adam(self.parameters(), lr=globals.learning_rate)

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
        features = F.relu(self.fc1(x))
        #value_features = F.relu(self.value_features(x))
        value_features = F.relu(self.value_features2(features))
        value = self.value(value_features)

        #advantages_features = F.relu(self.advantage_features(x))
        advantages_features = F.relu(self.advantage_features2(features))
        advantages = self.advantages(advantages_features)

        centered_advantages = advantages - advantages.mean(1).unsqueeze(1)
        return value + centered_advantages

    def get_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            q = self.forward(torch.from_numpy(state).view(
                1, self.state_space_size).float().to(globals.device))
            return torch.max(q, 1)[1].data[0]
