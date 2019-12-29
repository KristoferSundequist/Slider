import numpy as np
import torch


class Node:
    def __init__(self, inner_state: torch.Tensor, action_space_size: int,
                 policy: torch.Tensor):
        self.inner_state = inner_state
        self.action_space_size = action_space_size

        self.visit_counts = [0 for _ in range(action_space_size)]
        self.mean_values = [0 for _ in range(action_space_size)]
        self.policy = policy
        self.rewards = [None for _ in range(action_space_size)]

        self.edges = [None for _ in range(action_space_size)]

    def upper_confidence_bound(self, a: int):
        c1 = 1.25
        c2 = 19652

        visit_count_sum = np.sum(self.visit_counts)
        ucb = self.mean_values[a] + self.policy[a] * (np.sqrt(visit_count_sum) / (1 + self.visit_counts[a])) * \
            (c1 * np.log((visit_count_sum + c2 + 1) / c2))
        return ucb

    def get_action(self):
        ucbs = [
            self.upper_confidence_bound(a) for a in range(action_space_size)
        ]
        return np.argmax(ucbs)
    
    def expand(self, a: int):
        # r, s2 = dynamics(self.inner_state, a)
        # p, v = prediction(s2)
        # self.rewards[a] = r
        # self.edges[a] = Node(s2, self.action_space_size, p)
        pass

