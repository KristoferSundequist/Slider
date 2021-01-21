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


def get_episode(
        num_initial_states: int,
        max_iters: int,
        representation: Representation,
        dynamics: Dynamics,
        prediction: Prediction,
        temperature: float,
        game,
        discount: float):

    episode = Episode(game.state_space_size)

    initial_states = [np.zeros(game.state_space_size)
                      for i in range(num_initial_states)]

    for i in range(max_iters):
        state = game.get_state()

        initial_states.pop(0)
        initial_states.append(state)

        root = MCTS(initial_states, representation, dynamics, prediction, game.action_space_size, 50, .99)

        action = sample_action(root, temperature)
        reward, _ = game.step(action)

        episode.add_transition(reward, action, state, root.get_search_policy(), root.search_value())

    episode.calc_targets_gae(discount)
    return episode


def get_episodes(
        n_episodes: int,
        num_initial_states: int,
        max_iters: int,
        representation: Representation,
        dynamics: Dynamics,
        prediction: Prediction,
        temperature: float,
        gameFactory,
        discount):

    torch.set_num_threads(1)
    pool = Pool(ncpus)

    episodes = pool.starmap(get_episode,
                            [(num_initial_states, max_iters, representation, dynamics, prediction, temperature, gameFactory(), discount) for _ in range(n_episodes)])

    torch.set_num_threads(ncpus)
    
    return episodes


'''

TEST get_episode()

'''


def test_get_episode():
    game = slider.GameSimple(1000, 1000)

    num_initial_states = 7
    state_space_size = game.state_space_size
    inner_size = 30
    action_space_size = game.action_space_size
    representation = Representation(num_initial_states, state_space_size, inner_size)
    dynamics = Dynamics(inner_size, action_space_size)
    prediction = Prediction(inner_size, action_space_size)

    num_iterations = 30
    e = get_episode(num_initial_states, num_iterations, representation, dynamics, prediction, 1, game, 0.99)

    assert len(e.states) == num_iterations


'''

TEST get_episodes()

'''


def test_get_episodeS():
    def gameFactory(): return slider.GameSimple(1000, 1000)
    game = gameFactory()
    num_initial_states = 7
    state_space_size = game.state_space_size
    inner_size = 30
    action_space_size = game.action_space_size
    representation = Representation(num_initial_states, state_space_size, inner_size)
    dynamics = Dynamics(inner_size, action_space_size)
    prediction = Prediction(inner_size, action_space_size)

    num_iterations = 30
    num_episodes = 10
    episodes = get_episodes(10, num_initial_states, num_iterations, representation,
                            dynamics, prediction, 1, gameFactory, 0.99)

    assert len(episodes) == num_episodes
    assert len(episodes[0].states) == num_iterations

    assert not np.array_equal(episodes[0].states[0], episodes[1].states[0])  # make sure not same data in diff episodes
