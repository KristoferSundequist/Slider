import typing
import numpy as np
import random
from typing import *


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

    def update_value_and_policy(self, update_index: int, new_policy: [float], new_value: float):
        self._search_policies[update_index] = new_policy
        self._search_values[update_index] = new_value

    def add_transition(self, reward: float, action: int, state: np.ndarray, search_policy: [float], search_value: float):
        self._rewards.append(reward)
        self._actions.append(action)
        self._states.append(state)
        self._search_policies.append(search_policy)
        self._search_values.append(search_value)

    def calc_targets_gae(self, discount: float, lambd: float = 0.95):
        values = self._search_values.copy()
        values.append(0)

        targets = [0.0 for _ in range(len(self._rewards))]

        gae = 0
        for i in reversed(range(len(self._rewards))):
            delta = self._rewards[i] + discount*values[i+1] - values[i]
            gae = delta + discount*lambd*gae
            targets[i] = values[i] + gae

        self._value_targets = targets

    def calc_targets(self, discount: float):
        self._value_targets = [0.0 for _ in range(len(self._rewards))]
        R = 0
        for i in reversed(range(len(self._rewards))):
            R = self._rewards[i] + discount*R
            self._value_targets[i] = R

    # returns [(value, reward, action, search_policy, isNotDone)]

    def _make_targets(self, state_index: int, num_unroll_steps: int) \
            -> [(float, float, int, [float], bool)]:
        targets = []
        action_space_size = len(self._search_policies[0])
        for current_index in range(state_index, state_index + num_unroll_steps):
            if current_index < len(self._states):
                targets.append((self._value_targets[current_index], self._rewards[current_index], self._actions[current_index],
                                self._search_policies[current_index], True))
            else:
                targets.append((0, 0, np.random.randint(action_space_size),
                                [1.0/action_space_size for _ in range(action_space_size)], False))

        return targets

    # OLD WORKING VERSION
    def gather_initial_state_old(self, state_index: int, num_initial_states: int) -> [np.ndarray]:
        initial_states = [np.zeros(self._state_space_size)
                          for i in range(num_initial_states)]

        prev_index = max(0, state_index - num_initial_states)
        for i in range(prev_index+1, state_index+1):
            initial_states.pop(0)
            initial_states.append(self._states[i])

        return initial_states

    def gather_initial_state(self, state_index: int, num_initial_states: int, num_unroll_steps: int) -> List[np.ndarray]:
        initial_states = []

        start_index = state_index - num_initial_states + 1
        end_index = state_index + 1 + num_unroll_steps

        for i in range(start_index, end_index):
            if i >= 0:
                initial_states.append(self._states[i])
            else:
                initial_states.append(np.zeros(self._state_space_size))

        return initial_states

    def sample(self, num_initial_states: int, num_unroll_steps: int) \
            -> ([np.ndarray], [(float, float, int, [float], bool)]):

        state_index = np.random.randint(len(self._states)-num_unroll_steps)
        initial_states = self.gather_initial_state(state_index, num_initial_states, num_unroll_steps)
        targets = self._make_targets(state_index, num_unroll_steps)

        return (initial_states, targets)


'''

TEST gather_initial_state

'''


def testgather_initial_state():
    e = Episode(3)

    for i in range(100):
        e.add_transition(0, 2, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 0)

    initial_states = e.gather_initial_state(10, 4, 3)

    assert np.array_equal(initial_states, [np.array(
        [i, i, i]) for i in [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]])


def testgather_initial_state_with_zero_pad_for_start():
    e = Episode(3)

    for i in range(100):
        e.add_transition(0, 2, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 0)

    initial_states = e.gather_initial_state(4, 10, 5)

    assert np.array_equal(initial_states, [np.array([i, i, i])
                                           for i in [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])


'''

TEST sample()

'''


def test_sample():
    num_initial_states = 8
    num_unroll_steps = 20

    e = Episode(3)

    for i in range(25):
        e.add_transition(0, 2, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 0)

    e.calc_targets_gae(.99)
    # targets[i] : (value, reward, action, search_policy, isNotDone)
    (initial, targets) = e.sample(num_initial_states, num_unroll_steps)

    assert len(initial) == (num_initial_states + num_unroll_steps)
    assert len(targets) == num_unroll_steps
    assert targets[0][2] == 2


'''

    TEST _make_targets()

'''


def test_make_targets():
    e = Episode(3)

    for i in range(100):
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

    e.calc_targets(.99)

    # tagets == [(value, reward, action, search_policy, isNotDone)]
    targets = e._make_targets(85, 20)  # check out of bound
    for t in targets:
        print(t)
    assert len(targets) == 20  # correct num transitions
    assert targets[11][2] == 3  # correct action
    assert all([not targets[i][4]
               for i in range(15, 20)])  # end states are donestates
    assert targets[14][0] == 1  # correct value
    assert targets[13][0] == 1*0.99  # correct value w discount
    assert targets[12][0] == 1*0.99**2  # correct value w discount

    assert targets[13][1] == 0  # correct rewards
    assert targets[14][1] == 1  # correct rewards
    assert all([np.array_equal(targets[i][3], [85 + i, 0.2, 0.3, 0.4])
               for i in range(15)])


def test_make_targets_gae():
    e = Episode(3)

    for i in range(100):
        if i == 50:
            e.add_transition(1, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 0)
        elif i == 55:
            e.add_transition(0, 2, np.array([i, i, i]), [
                             i, 0.2, 0.3, 0.4], 0.5)
        elif i == 96:
            e.add_transition(0, 3, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 1)
        elif i == 97:
            e.add_transition(0, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 1)
        elif i == 98:
            e.add_transition(0, 2, np.array([i, i, i]), [
                             i, 0.2, 0.3, 0.4], 0.5)
        elif i == 99:
            e.add_transition(1, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 1)
        else:
            e.add_transition(0, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 0)

    discount = .99
    lambd = .95
    e.calc_targets_gae(discount, lambd)

    # tagets == [(value, reward, action, search_policy, isNotDone)]
    targets = e._make_targets(85, 20)  # check out of bound
    for t in targets:
        print(t)
    assert len(targets) == 20  # correct num transitions
    assert targets[11][2] == 3  # correct action
    assert all([not targets[i][4]
               for i in range(15, 20)])  # end states are donestates
    gae99 = (1 + discount*0 - 1)
    assert targets[14][0] == 1 + gae99
    gae98 = (0 + discount*1 - 0.5) + discount*lambd*gae99
    assert targets[13][0] == 0.5 + gae98
    gae97 = (0 + discount*0.5 - 1) + discount*lambd*gae98
    assert targets[12][0] == 1 + gae97

    assert targets[13][1] == 0  # correct rewards
    assert targets[14][1] == 1  # correct rewards
    assert all([np.array_equal(targets[i][3], [85 + i, 0.2, 0.3, 0.4])
               for i in range(15)])
