import numpy as np
import torch

from util import *


class Node:
    def __init__(self, inner_state: torch.Tensor, action_space_size: int,
                 policy: np.ndarray):
        self.inner_state = inner_state
        self.action_space_size = action_space_size

        self.visit_counts = [0 for _ in range(action_space_size)]
        self.mean_values = [0.0 for _ in range(action_space_size)]
        self.policy = policy
        self.rewards = [None for _ in range(action_space_size)]

        self.edges = [None for _ in range(action_space_size)]

    def get_mean_reward(self):
        none_null_rewards = [r for r in self.rewards if r is not None]
        if len(none_null_rewards) == 0:
            return None

        count = len(none_null_rewards)
        sum_r = sum(none_null_rewards)
        return sum_r/count

    def upper_confidence_bound(self, a: int, normalizer: Normalizer):
        c1 = 1.25
        c2 = 19652

        visit_count_sum = np.sum(self.visit_counts)
        ucb = normalizer.get_normalized(self.mean_values[a]) + self.policy[a] * (np.sqrt(visit_count_sum) / (1 + self.visit_counts[a])) * \
            (c1 + np.log((visit_count_sum + c2 + 1) / c2))
        return ucb

    def get_action(self, normalizer: Normalizer) -> int:
        ucbs = [self.upper_confidence_bound(a, normalizer)
                for a in range(self.action_space_size)]
        return np.argmax(ucbs)

    def get_search_policy(self, temperature=1) -> [float]:
        temperature = 1/temperature
        total_visits = sum(map(lambda v: v**temperature, self.visit_counts))
        probs = list(map(lambda k: (k**temperature) /
                     total_visits, self.visit_counts))
        return probs

    def search_value(self) -> float:
        p = self.get_search_policy()
        return sum([self.mean_values[a] * p[a] for a in range(self.action_space_size)])

    def expand(self, a: int, r: float, node):
        self.rewards[a] = r
        self.edges[a] = node

    def update(self, value: float, action: int):
        self.mean_values[action] = (
            self.visit_counts[action] * self.mean_values[action] + value) / (self.visit_counts[action] + 1)
        self.visit_counts[action] += 1


'''
    TEST UCB
'''


def test_ucb_one_visit_yields_highest_policy():
    n = Node(None, 4, np.array([0.2, 0.3, 0.4, 0.1]))
    n.visit_counts = [1, 0, 0, 0]
    a = n.get_action(Normalizer(0, 1))
    assert a == 2


def test_ucb_equal_visits_yields_highest_policy():
    n = Node(None, 4, np.array([0.2, 0.3, 0.4, 0.1]))
    n.visit_counts = [30, 30, 30, 30]
    a = n.get_action(Normalizer(0, 1))
    assert a == 2


def test_highest_policy_visited_alot_yields_other_action():
    n = Node(None, 4, np.array([0.2, 0.3, 0.4, 0.1]))
    n.visit_counts = [20, 20, 200, 20]
    a = n.get_action(Normalizer(0, 1))
    assert a == 1


def test_low_policy_action_getting_choosen_when_high_value():
    n = Node(None, 4, np.array([0.2, 0.3, 0.4, 0.1]))
    n.visit_counts = [20, 20, 20, 20]
    n.mean_values = [0.1, 0.2, 0.1, 1]
    a = n.get_action(Normalizer(0, 1))
    assert a == 3


def test_low_policy_action_getting_choosen_when_low_high_value():
    n = Node(None, 4, np.array([0.2, 0.3, 0.4, 0.1]))
    n.visit_counts = [20, 20, 20, 20]
    n.mean_values = [0.01, 0.02, 0.08, 0.1]
    a = n.get_action(Normalizer(0, 0.1))
    assert a == 3


def test_super_low_prob_dont_crash():
    n = Node(None, 4, np.array([1.0e-50, 0.3, 0.4, 0.2]))
    n.visit_counts = [1, 0, 0, 0]
    n.mean_values = [10, 2, 2, 2]
    a = n.get_action(Normalizer(0, 1))
    assert a == 0


'''

Test expand

'''


def test_expand():
    n = Node(None, 4, np.array([1.0e-50, 0.3, 0.4, 0.2]))
    n2 = Node(None, 4, np.array([1.0e-50, 0.3, 0.4, 0.2]))
    assert n.edges[2] == None
    n.expand(2, 0, n2)
    assert n.edges[2] == n2


'''
 
 Test update

'''


def test_update():
    n = Node(None, 4, np.array([1.0e-50, 0.3, 0.4, 0.2]))
    n2 = Node(None, 4, np.array([1.0e-50, 0.3, 0.4, 0.2]))
    assert n.edges[2] == None
    action1 = 2
    n.expand(action1, 0, n2)
    assert n.edges[action1] == n2
    assert n.visit_counts[action1] == 0
    n.update(10.0, action1)
    assert n.visit_counts[action1] == 1
    n.update(20.0, action1)
    assert n.visit_counts[action1] == 2
    n.update(36.0, action1)
    assert n.visit_counts[action1] == 3

    action2 = 0
    n.expand(action2, 0, n2)
    assert n.edges[action2] == n2
    assert n.visit_counts[action2] == 0
    n.update(9.0, action2)
    assert n.visit_counts[action2] == 1
    n.update(19.0, action2)
    assert n.visit_counts[action2] == 2
    n.update(35.0, action2)
    assert n.visit_counts[action2] == 3

    assert n.mean_values[action1] == 22.0
    assert n.mean_values[action2] == 21.0


'''

TEST VALUE

'''


def test_get_value():
    n = Node(None, 4, np.array([1.0e-50, 0.3, 0.4, 0.2]))
    n.mean_values = [0.23, 0.1, 0.54, 0.9]
    n.visit_counts = np.array([1, 1, 3, 5])
    assert n.search_value() == sum([0.23*0.1, 0.1*0.1, 0.54*0.3, 0.9*0.5])


'''

TEST SEARCH POLICY

'''


def test_get_search_policy():
    n = Node(None, 4, np.array([1.0e-50, 0.3, 0.4, 0.2]))
    n.visit_counts = np.array([1, 1, 3, 5])
    assert n.get_search_policy() == [.1, .1, .3, .5]


'''

    TEST GET MEAN REWARDS

'''


def test_get_mean_rewards():
    n = Node(None, 4, np.array([1.0e-50, 0.3, 0.4, 0.2]))
    n.rewards = [1, None, 2, 3.3]
    assert n.get_mean_reward() == 2.1


def test_get_mean_rewards_all_None():
    n = Node(None, 4, np.array([1.0e-50, 0.3, 0.4, 0.2]))
    n.rewards = [None, None, None, None]
    assert n.get_mean_reward() == None
