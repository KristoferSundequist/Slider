from typing import *
from episode import Episode
import torch
import torch.nn.functional as F
from dataclasses import dataclass

# CALCULATE RETURNS


def calculate_value_targets(rewards: List[float], values: List[float], discount_factor: float, lambd: float):
    targets = [0.0 for _ in range(len(rewards))]
    targets[-1] = values[-1]

    for i in reversed(range(len(values) - 1)):
        bootstrap = (1.0 - lambd) * values[i + 1] + lambd * targets[i + 1]
        targets[i] = rewards[i] + discount_factor * bootstrap
    return targets


def test_calculate_value_targets():
    # Arrange
    rewards = [0.0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0, 0, 0]

    # Act
    targets = calculate_value_targets(rewards, values, 0.99, 0.95)

    # Assert
    print(targets)
    assert True


# ARRANGE DATA
@dataclass
class TrainingData:
    states: torch.Tensor
    one_hot_actions: torch.Tensor
    values: torch.Tensor
    value_targets: torch.Tensor


def arrange_data(episodes: List[Episode], state_space_size: int, action_space_size: int):
    states = torch.tensor([e.states for e in episodes]).view(-1, state_space_size)
    actions = F.one_hot(torch.tensor([e.actions for e in episodes]).view(-1, 1).squeeze(), action_space_size)
    values = torch.tensor([e.values for e in episodes], dtype=torch.float).view(-1, 1).squeeze()
    value_targets = torch.tensor([e.value_targets for e in episodes], dtype=torch.float).view(-1, 1).squeeze()
    return TrainingData(states, actions, values, value_targets)


def test_arrange_data():
    # Arrange
    ep1 = Episode([[1.0, 2, 3], [4, 5, 6]], [0, 3], [0.8, 0.9], [0, 0], [0.95, 0.85], 0)
    ep2 = Episode([[7.0, 8, 9], [10, 11, 12]], [2, 1], [0.7, 0.6], [0, 1], [0.25, 0.35], 1)

    # Act
    arranged_data = arrange_data([ep1, ep2], 3, 4)

    # Assert
    assert arranged_data.states.size() == (4, 3)
    assert arranged_data.one_hot_actions.size() == (4, 4)
    assert arranged_data.values.size() == (4,)
    assert arranged_data.value_targets.size() == (4,)

