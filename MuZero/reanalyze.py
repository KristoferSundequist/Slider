import numpy as np
import torch
from multiprocessing import Pool, cpu_count
import scipy

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
        action_space_size: int,
        num_unroll_steps: int):
    
    scipy.random.seed()

    for i in range(episode.get_num_transitions()):
        initial_states = episode.gather_initial_state(i, num_initial_states, 0)

        root = MCTS(initial_states, representation, dynamics,
                    prediction, action_space_size, 50, discount)

        episode.update_value_and_policy(
            i, root.get_search_policy(), root.search_value())

    episode.calc_targets_gae(discount)
    return episode


def reanalze_episodes(
        episodes: [Episode],
        num_initial_states: int,
        representation: Representation,
        dynamics: Dynamics,
        prediction: Prediction,
        discount: float,
        action_space_size: int,
        num_unroll_steps: int):

    torch.set_num_threads(1)
    with Pool(ncpus) as pool:
        reanalyzed_episodes = pool.starmap(
            reanalyze_episode,
            [(episode, representation, dynamics, prediction, discount, num_initial_states,
              action_space_size, num_unroll_steps) for episode in episodes]
        )

    torch.set_num_threads(ncpus)

    return reanalyzed_episodes


'''

TEST test_something()

'''


def test_something():
    assert (5 == 5)
