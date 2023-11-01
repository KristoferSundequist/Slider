import torch
import globals
from typing import List

def combine_states(stoch_states: torch.Tensor, recurrent_states: torch.Tensor) -> torch.Tensor:
    return torch.concat([stoch_states, recurrent_states], 1)

def get_average_gradient(model: torch.nn.Module) -> float:
    grads = [param.grad.view(-1) for param in model.parameters()]
    return torch.cat(grads).abs().mean().item()

def calculate_value_targets_for_batch(rewards: torch.Tensor, values: torch.Tensor, lmbda: float = 0.95, discount: float = 0.99):
    targets = torch.zeros_like(values)
    targets[:, -1] = values[:, -1]

    for i in reversed(range(values.size()[1] - 1)):
        targets[:, i] = rewards[:, i] + discount * ((1.0 - lmbda) * values[:, i + 1] + lmbda * targets[:, i + 1])
    return targets


def calculate_value_targets_for_batch_old(
    rewards: torch.Tensor, values: torch.Tensor, discount_factor: float = 0.99, keep_value_ratio: float = 0.5
) -> torch.Tensor:
    value_targets = torch.zeros_like(rewards).float()
    value_targets[:, -1] = values[:, -1]

    discounted_rewards = torch.zeros_like(rewards).float()
    discounted_rewards[:, -1] = values[:, -1]

    for i in reversed(range(rewards.size()[1] - 1)):
        discounted_rewards[:, i] = rewards[:, i] + discount_factor * discounted_rewards[:, i + 1]
        value_targets[:, i] = (1 - keep_value_ratio) * discounted_rewards[:, i] + keep_value_ratio * values[:, i]

    return value_targets


def test_calculate_value_targets_for_batch_old():
    # Arrange
    rewards = torch.Tensor([[0.0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0.0, 0, 0, 0, 0, 0, 0, 0, 2, 0]])
    values = torch.Tensor(
        [[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0], [0.2, 5, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0]]
    )

    # Act
    targets = calculate_value_targets_for_batch_old(rewards, values, 0.99, 0.3)

    # Assert
    print(targets)
    assert targets.size() == rewards.size()
