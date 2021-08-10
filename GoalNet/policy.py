import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import torch.distributions
import numpy as np
import copy

'''''''''''''''''''''''''''''''''''''''
                VALUE NET
'''''''''''''''''''''''''''''''''''''''


class ValueNetwork(nn.Module):
    def __init__(self, state_size: int):
        super(ValueNetwork, self).__init__()
        self.hidden_size = 256

        self.input_layer = nn.Linear(state_size * 2, self.hidden_size)
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, 1)

        self.apply(self.weight_init)
        self.opt = optim.Adam(self.parameters(), lr=1e-4)

    def copy_weights(self, other):
        self.load_state_dict(copy.deepcopy(other.state_dict()))

    def save_weights(self, name):
        torch.save(self.state_dict(), name)

    def load_weights(self, name):
        self.load_state_dict(torch.load(name))
    
    def soft_update(self, source, tau):
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            size = m.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            variance = np.sqrt(2.0/(fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)

    def forward(self, states: torch.tensor, goals: torch.tensor):
        joint = torch.cat((states, goals), 1)
        x = F.relu(self.input_layer(joint))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x


def test_value_shape():
    vn = ValueNetwork(8)
    states = torch.rand(3, 8)
    goals = torch.rand(3, 8)
    values = vn.forward(states, goals)
    assert(values.size() == torch.Size([3, 1]))


'''''''''''''''''''''''''''''''''''''''
                POLICY NET
'''''''''''''''''''''''''''''''''''''''


class PolicyNetwork(nn.Module):
    def __init__(self, state_size: int, output_size: int):
        super(PolicyNetwork, self).__init__()
        self.hidden_size = 256

        self.input_layer = nn.Linear(state_size * 2, self.hidden_size)
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, output_size)

        self.apply(self.weight_init)
        self.opt = optim.Adam(self.parameters(), lr=1e-4)

    def copy_weights(self, other):
        self.load_state_dict(copy.deepcopy(other.state_dict()))

    def save_weights(self, name):
        torch.save(self.state_dict(), name)

    def load_weights(self, name):
        self.load_state_dict(torch.load(name))
    
    def soft_update(self, source, tau):
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            size = m.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            variance = np.sqrt(2.0/(fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)

    def forward(self, states: torch.tensor, goals: torch.tensor):
        joint = torch.cat((states, goals), 1)
        x = F.relu(self.input_layer(joint))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

    def get_action(self, state: np.array, goal: np.array) -> int:
        with torch.no_grad():
            logits = self.forward(
                torch.from_numpy(state).unsqueeze(0).float(),
                torch.from_numpy(goal).unsqueeze(0).float()
            )
            probs = F.softmax(logits, 1)
            action = Categorical(probs).sample().item()
            return action


def test_policy_shape():
    pn = PolicyNetwork(8, 4)
    states = torch.rand(3, 8)
    goals = torch.rand(3, 8)
    logits = pn.forward(states, goals)
    assert(logits.size() == torch.Size([3, 4]))


def test_get_action():
    pn = PolicyNetwork(8, 4)
    state = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    goal = np.array([-5, 2, 3, 4, 2, 6, 7, 3])
    action = pn.get_action(state, goal)
    assert(0 <= action and action <= 3)


'''''''''''''''''''''''''''''''''''''''
                GOAL NET
'''''''''''''''''''''''''''''''''''''''


class GoalNetwork(nn.Module):
    def __init__(self, state_size: int):
        super(GoalNetwork, self).__init__()
        self.hidden_size = 256

        self.input_layer = nn.Linear(state_size, self.hidden_size)
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, state_size)

        self.apply(self.weight_init)
        self.opt = optim.Adam(self.parameters(), lr=1e-4)

    def copy_weights(self, other):
        self.load_state_dict(copy.deepcopy(other.state_dict()))

    def save_weights(self, name):
        torch.save(self.state_dict(), name)

    def load_weights(self, name):
        self.load_state_dict(torch.load(name))
    
    def soft_update(self, source, tau):
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            size = m.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            variance = np.sqrt(2.0/(fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)

    def forward(self, states: torch.tensor):
        x = torch.tanh(self.input_layer(states))
        x = torch.tanh(self.hidden_layer(x))
        x = torch.tanh(self.output_layer(x))
        return x

    def get_goal(self, state: np.array) -> np.array:
        with torch.no_grad():
            return self.forward(torch.from_numpy(state).unsqueeze(0).float()).numpy()[0, :]


def test_goal_shape():
    gn = GoalNetwork(8)
    states = torch.rand(3, 8)
    goals = gn.forward(states)
    assert(goals.size() == torch.Size([3, 8]))


def test_get_goal():
    gn = GoalNetwork(7)
    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    goal = gn.get_goal(state)
    assert(goal.shape == (7,))
