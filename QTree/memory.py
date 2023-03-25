from collections import namedtuple
import numpy as np
import random
from dataclasses import dataclass
from typing import List

@dataclass
class Transition:
    state: List[float]
    action: int
    next_state: List[float]
    reward: float

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory: List[Transition] = []
        self.position = 0

    def push(self, transition: Transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def get_memories(self):
        return self.memory
