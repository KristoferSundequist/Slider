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
        self.value_target = None

    def add_transition(self, reward: float, action: int, state: np.ndarray, search_policy: [float], search_value: float):
        self.rewards.append(reward)
        self.actions.append(action)
        self.states.append(state)
        self.search_policies.append(search_policy)
        self.search_values.append(search_value)

    # returns [(value, reward, action, search_policy, isNotDone)]
    def make_targets(self, state_index: int, num_unroll_steps: int, discount: float, bootstrap_steps: int = 10) \
            -> [(float, float, int, [float], bool)]:

        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        action_space_size = len(self.search_policies[0])
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps):
            bootstrap_index = current_index + bootstrap_steps
            if bootstrap_index < len(self.states):
                value = self.search_values[bootstrap_index] * discount**bootstrap_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * discount**i  # pytype: disable=unsupported-operands

            if current_index < len(self.states):
                targets.append((value, self.rewards[current_index], self.actions[current_index],
                                self.search_policies[current_index], True))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, 0, np.random.randint(action_space_size),
                                [1.0/action_space_size for _ in range(action_space_size)], False))
        return targets


    def calc_targets(self, discount: float):
        self.value_target = np.zeros_like(self.rewards, dtype=np.float)
        R = 0
        for i in reversed(range(len(self.rewards))):
            R = self.rewards[i] + discount*R
            self.value_target[i] = R
        

    # returns [(value, reward, action, search_policy, isNotDone)]
    def make_targets_new(self, state_index: int, num_unroll_steps: int) \
            -> [(float, float, int, [float], bool)]:
        targets = []
        action_space_size = len(self.search_policies[0])
        for current_index in range(state_index, state_index + num_unroll_steps):
            if current_index < len(self.states):
                targets.append((self.value_target[current_index], self.rewards[current_index], self.actions[current_index],
                                self.search_policies[current_index], True))
            else:
                targets.append((0, 0, np.random.randint(action_space_size),
                                [1.0/action_space_size for _ in range(action_space_size)], False))

        return targets

    def gather_initial_state(self, state_index: int, num_initial_states: int) -> [np.ndarray]:
        initial_states = [np.zeros(self.state_space_size)
                          for i in range(num_initial_states)]

        prev_index = max(0, state_index - num_initial_states)
        for i in range(prev_index+1, state_index+1):
            initial_states.pop(0)
            initial_states.append(self.states[i])

        return initial_states

    def sample(self, num_initial_states: int, num_unroll_steps: int, discount: float) \
            -> ([np.ndarray], [(float, float, int, [float], bool)]):

        state_index = np.random.randint(len(self.states))
        initial_states = self.gather_initial_state(state_index, num_initial_states)
        targets = self.make_targets_new(state_index, num_unroll_steps)

        return (initial_states, targets)


'''

TEST gather_initial_state

'''


def test_gather_initial_state():
    e = Episode(3)

    for i in range(100):
        e.add_transition(0, 2, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 0)

    initial_states = e.gather_initial_state(10, 4)

    assert np.array_equal(initial_states, [np.array([i, i, i]) for i in [7.0, 8.0, 9.0, 10.0]])


def test_gather_initial_state_with_zero_pad_for_start():
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
        elif i == 96:
            e.add_transition(0, 3, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 3)
        elif i == 98:
            e.add_transition(0, 2, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 3)
        elif i == 99:
            e.add_transition(1, 2, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 1)
        else:
            e.add_transition(0, 2, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 0)

    targets = e.make_targets(85, 20, 0.99)  # check out of bound
    for t in targets:
        print(t)
    assert len(targets) == 20
    assert targets[11][2] == 3
    assert all([not targets[i][4] for i in range(15, 20)])  # end states are donestates
    assert True


'''

TEST sample()

'''


def test_sample():
    num_initial_states = 8
    num_unroll_steps = 20

    e = Episode(3)

    for i in range(100):
        e.add_transition(0, 2, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 0)

    e.calc_targets(.99)
    # targets[i] : (value, reward, action, search_policy, isNotDone)
    (initial, targets) = e.sample(num_initial_states, num_unroll_steps, .99)

    assert len(initial) == num_initial_states
    assert len(targets) == num_unroll_steps
    assert targets[0][2] == 2


'''

    TEST make_targets_new()

'''

def test_make_targets_new():
    e = Episode(3)

    for i in range(100):
        if i == 50:
            e.add_transition(1, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 0)
        elif i == 55:
            e.add_transition(0, 2, np.array([i, i, i]), [i, 0.2, 0.3, 0.4], 0.5)
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
    targets = e.make_targets_new(85, 20)  # check out of bound
    for t in targets:
        print(t)
    assert len(targets) == 20 #correct num transitions
    assert targets[11][2] == 3 #correct action
    assert all([not targets[i][4] for i in range(15, 20)])  # end states are donestates
    assert targets[14][0] == 1 #correct value
    assert targets[13][0] == 1*0.99 #correct value w discount
    assert targets[12][0] == 1*0.99**2 #correct value w discount

    assert targets[13][1] == 0 #correct rewards
    assert targets[14][1] == 1 #correct rewards
    assert all([np.array_equal(targets[i][3],[85 + i,0.2,0.3,0.4]) for i in range(15)])