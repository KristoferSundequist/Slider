import numpy as np
import random


class Episode:
    def __init__(self, state_space_size):
        self.rewards = []
        self.actions = []
        self.states = []
        self.search_policies = []
        self.search_values = []
        self.state_space_size = state_space_size

    def add_transition(self, reward: float, action: int, state: np.ndarray, search_policy: [float], search_value: float):
        self.rewards.append(reward)
        self.actions.append(action)
        self.states.append(state)
        self.search_policies.append(search_policy)
        self.search_values.append(search_value)

    def make_targets(self, state_index: int, num_unroll_steps: int, discount: float, bootstrap_steps: int = 10) -> [(float, float, [float], bool)]:
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + bootstrap_steps
            if bootstrap_index < len(self.states):
                value = self.search_values[bootstrap_index] * discount**bootstrap_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * discount**i  # pytype: disable=unsupported-operands

            if current_index < len(self.states):
                targets.append((value, self.rewards[current_index],
                                self.search_policies[current_index], False))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, 0, [], True))
        return targets

    def sample_batch(self, batch_size: int):
        inds = random.sample(range(len(self.states)), batch_size)
        batch = [(self.states[i], self.rewards[i], self.search_policies[i], self.search_values[i]) for i in inds]
        return batch

    def gather_initial_state(self, state_index: int, num_initial_states: int) -> [np.ndarray]:
        initial_states = [np.zeros(self.state_space_size)
                          for i in range(num_initial_states)]

        prev_index = max(0, state_index - num_initial_states)
        for i in range(prev_index+1, state_index+1):
            initial_states.pop(0)
            initial_states.append(self.states[i])

        return initial_states

    def sample(self):
        i = np.random.randint(len(self.states))
        return


'''

TEST gather_initial_state

'''


def test_gather_initial_state():
    e = Episode(3)

    for i in range(100):
        e.add_transition(0, 2, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 0)

    initial_states = e.gather_initial_state(10, 4)

    assert np.array_equal(initial_states, [np.array([i, i, i]) for i in [7.0, 8.0, 9.0, 10.0]])


def test_gather_initial_state_with_zero_pad():
    e = Episode(3)

    for i in range(100):
        e.add_transition(0, 2, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 0)

    initial_states = e.gather_initial_state(4, 10)

    assert np.array_equal(initial_states, [np.array([i, i, i])
                                           for i in [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]])


'''

TEST make_target

'''


def test_make_target():
    e = Episode(3)

    for i in range(100):
        if i == 50:
            e.add_transition(1, 2, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 0)
        elif i == 55:
            e.add_transition(0, 2, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 0.5)
        elif i == 100:
            e.add_transition(1, 2, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 0)
        else:
            e.add_transition(0, 2, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 0)
    
    targets = e.make_targets(43, 10, 0.99)
    for t in targets:
        print(t)
    assert True