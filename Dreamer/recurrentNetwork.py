import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import copy
import globals
from typing import *


class RecurrentnNetwork(nn.Module):
    def __init__(self, action_space_size: int):
        super(RecurrentnNetwork, self).__init__()
        self.action_space_size = action_space_size

        hidden_size = globals.mlp_size

        self.fc1 = nn.Linear(action_space_size + globals.stoch_vector_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.rnn = nn.GRUCell(hidden_size, globals.recurrent_vector_size)

        self.initial_hidden = nn.Parameter(torch.zeros(globals.recurrent_vector_size), requires_grad=True)

        self.apply(self.weight_init)
        self.opt = optim.AdamW(self.parameters(), lr=globals.world_model_learning_rate, weight_decay=0.001, eps=globals.world_model_adam_eps)

    def get_initial(self, n: int) -> torch.Tensor:
        return self.initial_hidden.repeat(n, 1)

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

    def forward(
        self, actions: torch.Tensor, prev_stoch_states: torch.Tensor, prev_hiddens: torch.Tensor
    ) -> torch.Tensor:
        assert actions.size()[1] == self.action_space_size
        assert prev_stoch_states.size()[1] == globals.stoch_vector_size
        assert prev_hiddens.size()[1] == globals.recurrent_vector_size
        concatted = torch.concat([prev_stoch_states, actions], 1)
        x = F.relu(self.fc1(concatted))
        x = F.relu(self.fc2(x))
        return self.rnn.forward(x, prev_hiddens)


def test_Recurrentn():
    # Arrange
    actions = torch.tensor([[0.0, 1, 0, 0], [0, 0, 0, 1]])
    prev_recurrent_states = torch.rand(2, globals.recurrent_vector_size)
    prev_stoch_states = torch.rand(2, globals.stoch_vector_size)
    rec = RecurrentnNetwork(4)

    # Act
    result = rec.forward(actions, prev_stoch_states, prev_recurrent_states)

    # Assert
    assert result.size() == (2, globals.recurrent_vector_size)
