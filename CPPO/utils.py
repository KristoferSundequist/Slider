from typing import *
from episode import Episode
import torch
import torch.nn.functional as F
from dataclasses import dataclass


# ARRANGE DATA
@dataclass
class TrainingData:
    states: torch.Tensor
    actions: torch.Tensor
    action_means: torch.Tensor
    action_stds: torch.Tensor
    value_logits: torch.Tensor
    value_targets: torch.Tensor


def arrange_data(
    episodes: List[Episode], state_space_size: int, action_space_size: int, discount_factor: float, lambd: float
):
    states = torch.tensor([e.states for e in episodes]).view(-1, state_space_size)
    actions = torch.tensor([e.actions for e in episodes]).view(-1, action_space_size)
    action_means = torch.tensor([e.action_means for e in episodes]).view(-1, action_space_size)
    action_stds = torch.tensor([e.action_stds for e in episodes]).view(-1, action_space_size)
    value_logits = torch.tensor([e.value_logits for e in episodes], dtype=torch.float).view(-1, 255)
    value_targets = (
        torch.tensor([e.get_value_targets(discount_factor, lambd) for e in episodes], dtype=torch.float)
        .view(-1, 1)
        .squeeze()
    )
    return TrainingData(states, actions, action_means, action_stds, value_logits, value_targets)


def test_arrange_data():
    # Arrange
    ep1 = Episode()
    ep1.add_transition([1.0, 2, 3], [0, 1], 0.8, 0, [0.0, 1.1], [-0.1, 0.1])
    ep1.add_transition([4.0, 5, 6], [2, 3], 0.9, 0, [2.2, 3.1], [1.2, 1.1])

    ep2 = Episode()
    ep2.add_transition([0.0, 2, 3], [-1, 2], 0.8, 1, [0.1, 1.1], [-0.3, 0.1])
    ep2.add_transition([4.0, 5, 6], [-2, 1], 0.5, 0, [2.2, 3.1], [1.2, 1.1])

    # Act
    arranged_data = arrange_data([ep1, ep2], 3, 2, 0.1, 0.1)

    # Assert
    assert arranged_data.states.size() == (4, 3)
    assert arranged_data.actions.size() == (4, 2)
    assert arranged_data.actions.tolist() == [[0, 1], [2, 3], [-1, 2], [-2, 1]]
    assert arranged_data.action_means.size() == (4, 2)
    assert arranged_data.action_means.equal(torch.tensor([[0.0, 1.1], [2.2, 3.1], [0.1, 1.1], [2.2, 3.1]]))
    assert arranged_data.action_stds.size() == (4, 2)
    assert arranged_data.action_stds.equal(torch.tensor([[-0.1, 0.1], [1.2, 1.1], [-0.3, 0.1], [1.2, 1.1]]))
    assert arranged_data.values.size() == (4,)
    assert arranged_data.values.equal(torch.tensor([0.8, 0.9, 0.8, 0.5]))
    assert arranged_data.value_targets.size() == (4,)
