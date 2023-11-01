import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import copy
import globals
from typing import *
from utils import get_average_gradient


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()

        hidden_size = globals.mlp_size
        self.fc1 = nn.Linear(globals.stoch_vector_size + globals.recurrent_vector_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.apply(self.weight_init)
        self.opt = optim.AdamW(
            self.parameters(),
            lr=globals.actor_critic_learning_rate,
            weight_decay=0.001,
            eps=globals.actor_critic_adam_eps,
        )

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
            m.weight.data.normal_(0.0, 0.1 * variance)
            m.bias.data.normal_(0.0, 0.001 * variance)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.size()[-1] == globals.stoch_vector_size + globals.recurrent_vector_size
        x = F.relu(self.fc1(hidden_states))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def test_value():
    # Arrange
    hiddens = torch.rand(7, globals.stoch_vector_size + globals.recurrent_vector_size)
    valueNetwork = ValueNetwork()

    # Act
    result = valueNetwork.forward(hiddens)

    # Assert
    assert result.size() == (7, 1)



####################################################################
########### VALUE HANDLER ##########################################
####################################################################

class ValueHandler:
    def __init__(self, logger):
        self._valueNetwork = ValueNetwork().to(globals.device)
        self._targetValueNetwork = copy.deepcopy(self._valueNetwork)
        self._logger = logger
        self._current_update_countdown = globals.target_network_gradient_steps
    
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self._valueNetwork.forward(values)
    
    def forward_lagged(self, values: torch.Tensor) -> torch.Tensor:
        return self._targetValueNetwork.forward(values)
    
    def update(self, loss: torch.Tensor):
        self._current_update_countdown -= 1
        if self._current_update_countdown <= 0:
            self._current_update_countdown = globals.target_network_gradient_steps
            self._targetValueNetwork = copy.deepcopy(self._valueNetwork)
        
        self._valueNetwork.opt.zero_grad()
        loss.backward()
        self._logger.add("Avg value grad", get_average_gradient(self._valueNetwork))
        max_norm_clip = 100
        nn.utils.clip_grad.clip_grad_value_(self._valueNetwork.parameters(), max_norm_clip)
        self._valueNetwork.opt.step()
        

