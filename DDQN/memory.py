from collections import namedtuple
import numpy as np
import random

class PrioritizedReplayMemory(object):

    def __init__(self, capacity):
        self.td_errors = np.zeros(capacity)
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.td_errors[self.position] = 1
        self.position = (self.position + 1) % self.capacity

    def update_td_errors(self, indices, errors):
        self.td_errors[indices] = errors

    def sample(self, batch_size, alpha):
        subsample_size = 500
        subsample = random.sample(range(self.capacity), subsample_size)
        
        exp_errors = np.power(self.td_errors[subsample]+1e-4, alpha)
        probs = exp_errors/exp_errors.sum()
        sub_indices = np.random.choice(subsample_size, batch_size, p=probs, replace=False)

        indices = [subsample[i] for i in sub_indices]
        
        return [self.memory[i] for i in indices], indices, [subsample_size*probs[i] for i in sub_indices]
        


    
'''
ReplayMemory class from:
pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''
        
Transition = namedtuple('Transition',
                    ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
