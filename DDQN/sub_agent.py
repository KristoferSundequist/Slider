import policy
import globals
from typing import *
import torch
import numpy as np
import memory
from transition import *
import torch.nn.functional as F
import math


class Sub_Agent:
    def __init__(self, state_space_size: int, action_space_size: int):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

        self.agent = policy.Policy(state_space_size, action_space_size)
        self.agent.to(globals.device)

        self.lagged_agent = policy.Policy(state_space_size, action_space_size)
        self.lagged_agent.copy_weights(self.agent)

        self.lagged_agent.to(globals.device)

    def reset(self):
        self.agent = policy.Policy(
            self.state_space_size, self.action_space_size)
        self.agent.to(globals.device)

        self.lagged_agent = policy.Policy(
            self.state_space_size, self.action_space_size)
        self.lagged_agent.copy_weights(self.agent)

        self.lagged_agent.to(globals.device)

    def forward(self, states: torch.Tensor):
        return self.agent.forward(states)

    def forward_target(self, states: torch.Tensor):
        return self.lagged_agent.forward(states)

    def get_action(self, state: np.ndarray) -> int:
        return self.agent.get_action(state)

    def get_values(self, state: np.ndarray) -> torch.tensor:
        return self.agent.forward(torch.from_numpy(state).view(
            1, self.state_space_size).float().to(globals.device))

    def update_target(self):
        self.lagged_agent.copy_weights(self.agent)

    def opt_step(self, loss):
        self.agent.opt.zero_grad()
        loss.backward()
        for param in self.agent.parameters():
            param.grad.data.clamp_(-1, 1)
        self.agent.opt.step()
