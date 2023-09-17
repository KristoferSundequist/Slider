from typing import *
from episode import Episode
from dataclasses import dataclass

# CALCULATE RETURNS


def calculate_value_targets(
    rewards: List[float], actions: List[int], values: List[List[float]], discount_factor: float, lambd: float
):
    targets = [0.0 for _ in range(len(rewards))]
    targets[-1] = values[-1][actions[-1]]

    for i in reversed(range(len(values) - 1)):
        bootstrap = (1.0 - lambd) * values[i + 1][actions[i + 1]] + lambd * targets[i + 1]
        targets[i] = rewards[i] + discount_factor * bootstrap
    return targets


def test_calculate_value_targets():
    # Arrange
    rewards = [0.0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    actions = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    values = [
        [0, 1, 0.4],
        [0.1, 0.5],
        [0.1, 0.6],
        [0.1, 0.7],
        [0.1, 0.8],
        [0.1, 0.9],
        [0.1, 1],
        [0.1, 0],
        [0.1, 0],
        [0.1, 0],
    ]

    # Act
    targets = calculate_value_targets(rewards, actions, values, 0.99, 0.95)

    # Assert
    print(targets)
    assert True


# ARRANGE DATA
@dataclass
class TrainingData:
    states_and_actions: List[List[float]]
    value_targets: List[float]


def arrange_data(episodes: List[Episode], action_space_size: int):
    states_and_actions = [
        e.states[ei] + [ai] for e in episodes for ei in range(len(e.states)) for ai in range(action_space_size)
    ]
    value_targets = [
        e.value_targets[ei] if ai == e.actions[ei] else e.values[ei][ai]
        for e in episodes
        for ei in range(len(e.states))
        for ai in range(action_space_size)
    ]
    return TrainingData(states_and_actions, value_targets)


def test_arrange_data():
    # Arrange
    ep1 = Episode([[1.0, 2, 3], [4, 5, 6]], [0, 3], [0.8, 0.9], [0, 0], [0.95, 0.85], 0)
    ep2 = Episode([[7.0, 8, 9], [10, 11, 12]], [2, 1], [0.7, 0.6], [0, 1], [0.25, 0.35], 1)

    # Act
    arranged_data = arrange_data([ep1, ep2], 3, 4)

    # Assert
    assert len(arranged_data.states_and_actions) == 4
    assert len(arranged_data.states_and_actions[0]) == 4
    assert arranged_data.states_and_actions[1] == [4, 5, 6, 3]
    assert arranged_data.values == [0.8, 0.9, 0.7, 0.6]
    assert arranged_data.value_targets == [0.95, 0.85, 0.25, 0.35]
