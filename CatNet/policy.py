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
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, globals.num_discrete_buckets)
             for _ in range(action_space_size)]
        )

        self.apply(self.weight_init)
        self.opt = optim.AdamW(
            self.parameters(), lr=globals.learning_rate, weight_decay=0.1)

        self.categories = torch.tensor([float(
            i) for i in range(-globals.abs_max_dicrete_value, globals.abs_max_dicrete_value+1)]).to(globals.device)

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

    def forward_features(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return x

    def forward_logits(self, x) -> List[torch.Tensor]:
        features = self.forward_features(x)
        return [q_head(features) for q_head in self.q_heads]

    def get_expected_values(self, logits_by_action: List[torch.Tensor]) -> torch.Tensor:
        assert len(
            logits_by_action) == self.action_space_size, f'{len(logits_by_action)} == {self.action_space_size}'
        values = [torch.matmul(F.softmax(logits, 1), self.categories)
                  for logits in logits_by_action]
        return torch.stack(values).transpose(0, 1)

    def get_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            parsed_state = torch.from_numpy(state).view(
                1, self.state_space_size).float().to(globals.device)
            logits = self.forward_logits(parsed_state)
            expected_values = self.get_expected_values(logits)
            return torch.max(expected_values, 1)[1].data[0]

    def opt_step(self, loss):
        self.opt.zero_grad()
        loss.backward()
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.opt.step()
