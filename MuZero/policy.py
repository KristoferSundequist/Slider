import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
import numpy as np

'''

REPRESENTATION

'''


class Representation(nn.Module):
    def __init__(self, num_states: int, state_space_size: int,
                 inner_size: int):
        super(Representation, self).__init__()

        self.state_space_size = state_space_size

        self.w1 = nn.Linear(num_states * state_space_size, 200)
        self.w2 = nn.Linear(200, inner_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.w1(input))
        out = self.w2(x)
        return out


def test_repr_forward_shape():
    num_states = 3
    state_space_size = 10
    inner_size = 20
    batch_size = 7

    r = Representation(num_states, state_space_size, inner_size)
    real_states = torch.rand(batch_size, num_states * state_space_size)
    inner_states = r.forward(real_states)
    assert inner_states.shape == torch.Size([batch_size, inner_size])


'''

DYNAMICS

'''


class Dynamics(nn.Module):
    def __init__(self, inner_size: int, action_space_size: int):
        super(Dynamics, self).__init__()

        self.inner_size = inner_size
        self.action_space_size = action_space_size

        self.w1 = nn.Linear(inner_size + action_space_size, 200)

        self.reward_out = nn.Linear(200, 1)
        self.state_out = nn.Linear(200, inner_size)

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        joint = torch.cat((state, action), 1)
        x = F.relu(self.w1(joint)) # Probably change to recurrent layer later
        reward = self.reward_out(x)
        next_state = self.state_out(x)
        return reward, next_state


def test_dynamics_forward_shape():
    inner_size = 20
    action_space_size = 4
    batch_size = 7

    d = Dynamics(inner_size, action_space_size)

    prev_states = torch.rand(batch_size, inner_size)
    actions = torch.rand(batch_size, action_space_size)

    rewards, next_states = d.forward(prev_states, actions)
    assert rewards.shape == torch.Size([batch_size, 1])
    assert next_states.shape == torch.Size([batch_size, inner_size])



'''

PREDICTION

'''


class Prediction(nn.Module):
    def __init__(self, inner_size: int, action_space_size: int):
        super(Prediction, self).__init__()

        self.inner_size = inner_size
        self.action_space_size = action_space_size

        self.w1 = nn.Linear(inner_size, 200)

        self.policy_out = nn.Linear(200, action_space_size)
        self.value_out = nn.Linear(200, 1)

    def forward(self, state: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = F.relu(self.w1(state)) # Probably change to recurrent layer later
        policy = F.softmax(self.policy_out(x), dim=1)
        value = self.value_out(x)
        return policy, value #softmax


def test_prediction_forward_shape():
    inner_size = 20
    action_space_size = 4
    batch_size = 7

    p = Prediction(inner_size, action_space_size)

    prev_states = torch.rand(batch_size, inner_size)

    policy, value = p.forward(prev_states)
    assert policy.shape == torch.Size([batch_size, action_space_size])
    assert value.shape == torch.Size([batch_size, 1])

######################

'''
INTEGRATION TEST
'''

def test_together_shapes():
    num_initial_states = 13
    state_space_size = 6
    inner_size = 20
    action_space_size = 4
    batch_size = 7

    r = Representation(num_initial_states, state_space_size, inner_size)
    d = Dynamics(inner_size, action_space_size)
    p = Prediction(inner_size, action_space_size)

    initial_states = torch.rand(batch_size, num_initial_states * state_space_size)
    initial_inner_states = r.forward(initial_states)

    actions = torch.rand(batch_size, action_space_size)
    rewards1, second_inner_states = d.forward(initial_inner_states, actions)
    actions2 = torch.rand(batch_size, action_space_size)
    rewards2, third_inner_states = d.forward(second_inner_states, actions2)

    policy, value = p.forward(third_inner_states)

    assert policy.shape == torch.Size([batch_size, action_space_size])
    assert value.shape == torch.Size([batch_size, 1])
    assert rewards2.shape == torch.Size([batch_size, 1])
    assert third_inner_states.shape == torch.Size([batch_size, inner_size])




