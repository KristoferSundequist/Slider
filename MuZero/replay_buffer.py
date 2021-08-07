import numpy as np
import random
from episode import *


class Replay_buffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.replay_buffer = []

    def sample_batch(self, batch_size: int, num_initial_states: int, num_unroll_steps: int):
        inds = [np.random.randint(len(self.replay_buffer)) for _ in range(batch_size)]
        batch = [self.replay_buffer[i].sample(num_initial_states, num_unroll_steps) for i in inds]
        return batch
    
    def sample_episodes(self, n_episodes: int) -> ([int], [Episode]):
        inds = random.sample(range(len(self.replay_buffer)), n_episodes)
        episodes = [self.replay_buffer[i] for i in inds]
        return inds, episodes

    def add_episode(self, e: Episode):
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(e)

    def replace_episode(self, i: int, e: Episode):
        self.replay_buffer[i] = e

'''

    TEST add_episode

'''


def test_add_episode():
    buffer_size = 3
    replay_buffer = Replay_buffer(buffer_size)
    assert len(replay_buffer.replay_buffer) == 0
    replay_buffer.add_episode(Episode(10))
    assert len(replay_buffer.replay_buffer) == 1
    replay_buffer.add_episode(Episode(10))
    assert len(replay_buffer.replay_buffer) == 2
    replay_buffer.add_episode(Episode(10))
    assert len(replay_buffer.replay_buffer) == buffer_size
    replay_buffer.add_episode(Episode(10))
    assert len(replay_buffer.replay_buffer) == buffer_size
    replay_buffer.add_episode(Episode(10))
    assert len(replay_buffer.replay_buffer) == buffer_size


'''

    TEST sample_batch

'''

def test_sample_batch():
    rb = Replay_buffer(10)
    for i in range(10):
        e = Episode(3)
        for i in range(100):
            e.add_transition(0, i, np.array([i, i, i]), [0.1, 0.2, 0.3, 0.4], 0)
        e.calc_targets(.99)
        rb.add_episode(e)
    
    batch = rb.sample_batch(32, 7, 5)
    assert len(batch) == 32
    
    (initial_states, targets) = batch[0]
    assert len(initial_states) == 7
    assert len(targets) == 5