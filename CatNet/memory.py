import numpy as np
import random
from typing import *
from transition import *

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory: List[Transition] = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
