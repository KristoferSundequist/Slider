import numpy as np
import random
from typing import List
from sequenceBuffer import Sequence

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory: List[Sequence] = []
        self.position = 0

    def push(self, sequence: Sequence):
        if len(self.memory) < self.capacity:
            self.memory.append(sequence)
        else:
            self.memory[self.position] = sequence
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size) -> List[Sequence]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def test_replay_buffer():
    memory = ReplayBuffer(3)
    memory.push(Sequence([], [1], []))
    memory.push(Sequence([], [2], []))
    memory.push(Sequence([], [3], []))
    memory.push(Sequence([], [4], []))

    assert memory.memory[0].actions[0] == 4
    assert memory.memory[1].actions[0] == 2
    assert memory.memory[2].actions[0] == 3
    assert len(memory.memory) == 3
    