import math
import numpy as np


class Nstep(object):

    def __init__(self, n):
        self.n = n
        self.states = []
        self.actions = []
        self.rewards = []
        self.goals = []

    def push(self, state: np.array, action: int, reward: float, goal: np.array):
        if len(self.states) >= self.n:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.goals.pop(0)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.goals.append(goal)

    def get(self):
        reward = 0
        for i in range(self.n):
            reward += math.pow(0.99, i)*self.rewards[i]
        return self.states[0], self.actions[0], self.states[self.n-1], reward, self.goals[0], self.goals[self.n-1]
