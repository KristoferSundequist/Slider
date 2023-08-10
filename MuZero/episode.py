import typing
import numpy as np
import random
from typing import *
from dataclasses import dataclass


@dataclass
class TargetValues:
    value: float
    reward: float
    action: int
    search_policy: List[float]


@dataclass
class TrainingExample:
    initialStates: List[np.ndarray]
    targetValues: List[TargetValues]


class Episode:
    def __init__(self, state_space_size):
        self._rewards: List[float] = []
        self._actions: List[int] = []
        self._states: List[np.ndarray] = []
        self._search_policies: List[list[float]] = []
        self._search_values: List[float] = []
        self._state_space_size: int = state_space_size
        self._value_targets: List[float] | None = None

    def get_reward_sum(self) -> float:
        return sum(self._rewards)

    def get_num_transitions(self) -> int:
        return len(self._states)

    def update_value_and_policy(self, update_index: int, new_policy: List[float], new_value: float):
        self._search_policies[update_index] = new_policy
        self._search_values[update_index] = new_value

    def add_transition(self, reward: float, action: int, state: np.ndarray, search_policy: List[float], search_value: float):
        self._rewards.append(reward)
        self._actions.append(action)
        self._states.append(state)
        self._search_policies.append(search_policy)
        self._search_values.append(search_value)

    def _make_targets(self, start_index: int, num_unroll_steps: int) -> List[TargetValues]:
        assert start_index + num_unroll_steps < len(
            self._states), 'start index + num_unrollsteps needs to be smaller than episode length'
        targets: List[TargetValues] = []
        for i in range(num_unroll_steps):
            current_index = start_index + i
            if current_index < len(self._states):
                targetValues = TargetValues(
                    value=self._value_targets[current_index],
                    reward=self._rewards[current_index],
                    action=self._actions[current_index],
                    search_policy=self._search_policies[current_index]
                )
                targets.append(targetValues)
            else:
                assert False, 'this should not happen'

        return targets

    def calculate_value_targets(self, discount_factor: float):
        targets = [0.0 for _ in range(len(self._rewards))]
        targets[-1] = self._search_values[-1]

        for i in reversed(range(len(self._rewards)-1)):
            targets[i] = self._rewards[i] + discount_factor * targets[i+1]
        
        self._value_targets = targets

    def _make_value_targets(self, start_index: int, num_unroll_steps: int, discount_factor: float) -> List[float]:
        assert start_index + num_unroll_steps < len(
            self._states), 'start index + num_unrollsteps needs to be smaller than episode length'
        value_targets = [0 for _ in range(num_unroll_steps)]
        value_targets[-1] = self._search_values[start_index +
                                                num_unroll_steps - 1]

        for i in reversed(range(num_unroll_steps-1)):
            value_targets[i] = self._rewards[start_index + i] + \
                discount_factor * value_targets[i + 1]

        return value_targets

    def gather_initial_state(self, start_index: int, num_initial_states: int) -> List[np.ndarray]:
        initial_states = [np.zeros(self._state_space_size)
                          for i in range(num_initial_states)]

        prev_index = max(0, start_index - num_initial_states)
        for i in range(prev_index+1, start_index+1):
            initial_states.pop(0)
            initial_states.append(self._states[i])

        return initial_states

    def sample(self, num_initial_states: int, num_unroll_steps: int) -> TrainingExample:

        start_index = np.random.randint(len(self._states) - num_unroll_steps)
        assert start_index >= 0
        initial_states = self.gather_initial_state(
            start_index, num_initial_states)
        targets = self._make_targets(
            start_index, num_unroll_steps)

        return TrainingExample(initialStates=initial_states, targetValues=targets)


'''

TEST gather_initial_state

'''


def testgather_initial_state():
    e = Episode(3)

    for i in range(100):
        e.add_transition(0, 2, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 0)

    initial_states = e.gather_initial_state(10, 4)

    assert len(initial_states) == 4
    assert np.array_equal(initial_states, [np.array(
        [i, i, i]) for i in [7.0, 8.0, 9.0, 10.0]])


def testgather_initial_state_with_zero_pad_for_start():
    e = Episode(3)

    for i in range(100):
        e.add_transition(0, 2, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 0)

    initial_states = e.gather_initial_state(4, 10)

    assert np.array_equal(
        initial_states,
        [np.array([i, i, i])
         for i in [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]]
    )


'''

TEST sample()

'''


def test_sample():
    num_initial_states = 8
    num_unroll_steps = 20

    e = Episode(3)

    for i in range(25):
        e.add_transition(0, 2, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 0)

    e.calculate_value_targets(discount_factor=.99)
    # targets[i] : (value, reward, action, search_policy, isNotDone)
    trainingExample = e.sample(num_initial_states, num_unroll_steps)

    print(len(trainingExample.initialStates))
    assert len(trainingExample.initialStates) == num_initial_states
    assert len(trainingExample.targetValues) == num_unroll_steps
    assert trainingExample.targetValues[0].action == 2


'''

    TEST _make_value_targets

'''


def test_make_value_targets():
    e = Episode(3)

    for i in range(100):
        if i == 50:
            e.add_transition(0, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 0)
        elif i == 51:
            e.add_transition(0, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 0)
        elif i == 52:
            e.add_transition(1, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 0)
        elif i == 53:
            e.add_transition(0, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 0)
        elif i == 54:
            e.add_transition(0, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 5)
        elif i == 55:
            e.add_transition(0, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 0)
        elif i == 56:
            e.add_transition(1, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 0)
        elif i == 57:
            e.add_transition(1, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 0)
        else:
            e.add_transition(0, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 0)

    # tagets == [(value, reward, action, search_policy, isNotDone)]
    value_targets = e._make_value_targets(50, 5, 0.99)  # check out of bound

    assert len(value_targets) == 5  # correct num transitions
    assert value_targets[4] == 5
    assert value_targets[3] == 5*0.99
    assert value_targets[2] == 5*0.99*0.99 + 1


'''

    TEST _make_targets()

'''


def test_make_targets():
    e = Episode(3)

    for i in range(110):
        if i == 50:
            e.add_transition(1, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 0)
        elif i == 55:
            e.add_transition(0, 2, np.array([i, i, i]), [
                             i, 0.2, 0.3, 0.4], 0.5)
        elif i == 96:
            e.add_transition(0, 3, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 3)
        elif i == 98:
            e.add_transition(0, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 3)
        elif i == 99:
            e.add_transition(1, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 1)
        else:
            e.add_transition(0, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 0)

    e.calculate_value_targets(.99)
    # tagets == [(value, reward, action, search_policy, isNotDone)]
    targets = e._make_targets(90, 10)  # check out of bound
    for t in targets:
        print(t)
    assert len(targets) == 10  # correct num transitions3
    assert targets[5].action == 2  # correct action
    assert targets[6].action == 3  # correct action
    assert targets[7].action == 2  # correct action
    assert all([np.array_equal(targets[i].search_policy, [90 + i, 0.2, 0.3, 0.4])
               for i in range(10)])
