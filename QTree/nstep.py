import math
import globals
from memory import Transition

class Nstep(object):

    def __init__(self, n):
        self.n = n
        self.states = []
        self.actions = []
        self.rewards = []

    def push(self, state, action, reward):
        if len(self.states) >= self.n:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def get(self) -> Transition:
        reward = 0
        for i in range(self.n):
            reward += math.pow(globals.discount_factor, i)*self.rewards[i]
        return Transition(self.states[0], self.actions[0], self.states[self.n-1], reward)
        
        
        
