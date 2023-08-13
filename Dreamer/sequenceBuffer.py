import math
import globals
from dataclasses import dataclass
from typing import List
import copy

@dataclass
class Sequence:
    observations: List[List[float]]
    actions: List[int]
    rewards: List[float]

class SequenceBuffer:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.observations: List[List[float]] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []

    def push(self, observation: List[float], action: int, reward: float):
        if len(self.observations) >= self.sequence_length:
            self.observations.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)

    def get(self) -> Sequence:
        assert len(self.observations) >= self.sequence_length
        return Sequence(copy.deepcopy(self.observations), copy.deepcopy(self.actions), copy.deepcopy(self.rewards))

def test_sequence_buffer():
    buffer = SequenceBuffer(3)
    for i in range(5):
        buffer.push([i], i, i)

    seq = buffer.get()
    assert seq.observations == [[2],[3],[4]]
    assert seq.actions == [2,3,4]
    assert seq.rewards == [2,3,4]