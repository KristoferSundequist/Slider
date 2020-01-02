import numpy as np

class Episode:
    def __init__(self):
        self.rewards = []
        self.actions = []
        self.states = []
    
    def add_transition(self, reward: float, action: int, state: np.ndarray):
        self.rewards.append(reward)
        self.actions.append(action)
        self.states.append(state)