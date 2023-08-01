import math
import globals

class Nstep(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def push(self, state, action, reward):
        if len(self.states) > globals.nsteps:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def get(self):
        reward = 0
        for i in range(globals.nsteps):
            reward += math.pow(globals.discount_factor, i)*self.rewards[i]
        return self.states[0], self.actions[0], self.states[globals.nsteps], reward
        
        
        
