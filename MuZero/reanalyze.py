import numpy as np
import policy
import torch
import time
import copy
from multiprocessing import Pool, cpu_count

import slider
from policy import *
from MCTS import *
from episode import *

ncpus = cpu_count()


def reanalyze_episode(
        episode: Episode,
        representation: Representation,
        dynamics: Dynamics,
        prediction: Prediction,
        discount: float,
        num_initial_states: int,
        action_space_size: int):

    for i in range(episode.get_num_transitions()):
        initial_states = episode.gather_initial_state(i, num_initial_states)

        root = MCTS(initial_states, representation, dynamics, prediction, action_space_size, 50, .99)

        episode.update_value_and_policy(i, root.get_search_policy(), root.search_value())

    episode.calc_targets_gae(discount)
    return episode


def reanalze_episodes(
        episodes: [Episode],
        num_initial_states: int,
        representation: Representation,
        dynamics: Dynamics,
        prediction: Prediction,
        discount: float,
        action_space_size: int):

    torch.set_num_threads(1)
    pool = Pool(ncpus)

    reanalyzed_episodes = pool.starmap(reanalyze_episode,
                            [(episode, representation, dynamics, prediction, discount, num_initial_states, action_space_size) for episode in episodes])

    torch.set_num_threads(ncpus)
    
    return reanalyzed_episodes


'''

TEST test_something()

'''

def test_something():
    assert(5 == 5)
