from lightgbm import LGBMRegressor
from typing import *
import numpy as np


class Agent:
    def __init__(self, state_space_size: int, action_space_size: int):
        self.model = LGBMRegressor(min_child_samples=2, num_leaves=2048, max_bin=255)
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.init_fit()

    def init_fit(self):
        self.model.fit(np.random.randn(10000, self.state_space_size + 1), np.random.randn(10000))

    def get_qvalues(self, state: List[float]):
        input_states = [state + [a] for a in range(self.action_space_size)]
        return self.model.predict(input_states)

    def fit(self, state_and_actions: List[List[float]], value_targets: List[float]):
        self.model.fit(state_and_actions, value_targets)
