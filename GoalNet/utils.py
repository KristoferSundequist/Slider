import torch
from typing import List
import pytest

'''

    Categotical cross entropy

'''


def categorical_cross_entropy(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()

def test_size_assert_categorical_cross_entropy_fail():
    with pytest.raises(AssertionError):
        result = categorical_cross_entropy(torch.tensor([[0.2,0.3]]), torch.tensor([[0.6,0.2,0.3]]))

def test_categorical_cross_entropy():
    a = torch.tensor([[0.2, 0.1, 0.5, 0.2]])
    b = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
    c = torch.tensor([[0.2, 0.2, 0.4, 0.2]])
    loss_ab = categorical_cross_entropy(a, b)
    loss_ac = categorical_cross_entropy(a, c)
    assert loss_ab > loss_ac


def test_categorical_cross_entropy():
    a = torch.tensor([[0.2, 0.1, 0.5, 0.2]])
    b = torch.tensor([[0.7, 0.1, 0.1, 0.1]])

    c = torch.tensor([[0.2, 0.2, 0.4, 0.2]])
    d = torch.tensor([[0.1, 0.1, 0.1, 0.7]])
    loss_ac = categorical_cross_entropy(a, c)
    loss_bd = categorical_cross_entropy(b, d)

    ab = torch.cat([a, b])
    cd = torch.cat([c, d])
    loss_ac_bd = categorical_cross_entropy(ab, cd)

    assert (loss_ac + loss_bd)/2 == loss_ac_bd


'''

    POLICY SETTUPER

'''


def get_policies_from_actions(actions: List[int], action_space_size: int, policy_upper_cap: float) -> torch.Tensor:
    policy_lower = (1.0 - policy_upper_cap) / (action_space_size - 1)
    return torch.FloatTensor(
        [
            [
                policy_upper_cap if i == action else policy_lower
                for i in range(action_space_size)
            ]
            for action in actions
        ]
    )


def test_get_policies_from_action():
    result = get_policies_from_actions([0, 3, 2, 1, 2], 5, 0.8)
    assert torch.equal(
        result,
        torch.FloatTensor([
            [0.8, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.8, 0.05],
            [0.05, 0.05, 0.8, 0.05, 0.05],
            [0.05, 0.8, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.8, 0.05, 0.05]
        ])
    )
