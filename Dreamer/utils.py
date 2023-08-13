import torch
import globals
from typing import List


def init_hidden(n: int):
    return torch.zeros(n, globals.hidden_vector_size)


def get_onehot(index: int, size: int):
    onehot = [0 for _ in range(size)]
    onehot[index] = 1
    return onehot


def test_onehot():
    assert get_onehot(3, 5) == [0, 0, 0, 1, 0]

def calculate_value_targets_for_batch(
    rewards: torch.Tensor, values: torch.Tensor, discount_factor: float, keep_value_ratio: float
) -> torch.Tensor:
    value_targets = torch.zeros_like(rewards).float()
    value_targets[:, -1] = values[:, -1]

    discounted_rewards = torch.zeros_like(rewards).float()
    discounted_rewards[:, -1] = values[:, -1]

    for i in reversed(range(rewards.size()[1] - 1)):
        discounted_rewards[:, i] = rewards[:, i] + discount_factor * discounted_rewards[:, i + 1]
        value_targets[:, i] = (1 - keep_value_ratio) * discounted_rewards[:, i] + keep_value_ratio * values[:, i]

    return value_targets


def test_calculate_value_targets_for_batch():
    # Arrange
    rewards = torch.Tensor([[0.0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0.0, 0, 0, 0, 0, 0, 0, 0, 2, 0]])
    values = torch.Tensor(
        [[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0], [0.2, 5, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0]]
    )

    # Act
    targets = calculate_value_targets_for_batch(rewards, values, 0.99, 0.3)

    # Assert
    print(targets)
    assert targets.size() == rewards.size()
