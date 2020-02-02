import numpy as np
import random
from episode import *


class Replay_buffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.replay_buffer = []

    def sample_batch(self, batch_size: int, num_initial_states: int, num_unroll_steps: int, discount: float):
        inds = [np.random.randint(len(self.replay_buffer)) for _ in range(batch_size)]
        batch = [self.replay_buffer[i].sample(num_initial_states, num_unroll_steps, discount) for i in inds]
        return batch

    def add_episode(self, e: Episode, discount):
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)
        e.calc_targets_gae(discount)
        self.replay_buffer.append(e)


'''

    TEST add_episode

'''


def test_add_episode():
    buffer_size = 3
    replay_buffer = Replay_buffer(buffer_size)
    assert len(replay_buffer.replay_buffer) == 0
    replay_buffer.add_episode(Episode(10), .99)
    assert len(replay_buffer.replay_buffer) == 1
    replay_buffer.add_episode(Episode(10), .99)
    assert len(replay_buffer.replay_buffer) == 2
    replay_buffer.add_episode(Episode(10), .99)
    assert len(replay_buffer.replay_buffer) == buffer_size
    replay_buffer.add_episode(Episode(10), .99)
    assert len(replay_buffer.replay_buffer) == buffer_size
    replay_buffer.add_episode(Episode(10), .99)
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
        rb.add_episode(e, .99)
    
    batch = rb.sample_batch(32, 7, 5, .99)
    assert len(batch) == 32
    
    (initial_states, targets) = batch[0]
    assert len(initial_states) == 7
    assert len(targets) == 5