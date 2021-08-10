from collections import namedtuple
import numpy as np
import random
from typing import List

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'goal', 'next_goal'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity: int = capacity
        self.memory: List[Transition] = []
        self.position: int = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def sample_trajectories(self, batch_size: int, trajectory_length: int) -> List[List[Transition]]:
        trajectories = []
        for _ in range(batch_size):
            index = random.randint(0, self.capacity-trajectory_length)
            trajectories.append(self.memory[index:(index+trajectory_length)])
        return trajectories

    def __len__(self) -> int:
        return len(self.memory)
