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
#width = 1500
#height = 1000
#num_initial_states = 7
#state_space_size = Game.state_space_size
#inner_size = 30
#action_space_size = Game.action_space_size
#representation = Representation(num_initial_states, state_space_size, inner_size)
#dynamics = Dynamics(inner_size, action_space_size)
#prediction = Prediction(inner_size, action_space_size)


def get_episode(
    game_width: int,
    game_height: int,
    num_initial_states: int,
    max_iters: int,
    representation: Representation,
    dynamics: Dynamics,
    prediction: Prediction):

    game = slider.Game(game_width, game_height)
    episode = Episode(game.state_space_size)

    initial_states = [np.zeros(game.state_space_size)
                      for i in range(num_initial_states)]

    for i in range(max_iters):
        state = game.get_state()

        initial_states.pop(0)
        initial_states.append(state)
        
        root = MCTS(initial_states, representation, dynamics, prediction, game.action_space_size, 50, .99)

        action = sample_action(root)
        reward,_ = game.step(action)

        episode.add_transition(reward, action, state, root.get_search_policy(), root.value())
    
    return episode

def get_episodes(
    n_episodes: int,
    game_width: int,
    game_height: int,
    num_initial_states: int,
    max_iters: int,
    representation: Representation,
    dynamics: Dynamics,
    prediction: Prediction):

    torch.set_num_threads(1)
    pool = Pool(ncpus)
    torch.set_num_threads(ncpus)
    
    args = (game_width, game_height, num_initial_states, max_iters, representation, dynamics, prediction)
    episodes = pool.starmap(get_episode, [args for _ in range(n_episodes)])

    return episodes





'''

TEST get_episode()

'''

def test_get_episode():
    num_initial_states = 7
    state_space_size = slider.Game.state_space_size
    inner_size = 30
    action_space_size = slider.Game.action_space_size
    representation = Representation(num_initial_states, state_space_size, inner_size)
    dynamics = Dynamics(inner_size, action_space_size)
    prediction = Prediction(inner_size, action_space_size)

    num_iterations = 30
    e = get_episode(1000, 1000, num_initial_states, num_iterations, representation, dynamics, prediction)

    assert len(e.states) == num_iterations

'''

TEST get_episodes()

'''

def test_get_episodeS():
    num_initial_states = 7
    state_space_size = slider.Game.state_space_size
    inner_size = 30
    action_space_size = slider.Game.action_space_size
    representation = Representation(num_initial_states, state_space_size, inner_size)
    dynamics = Dynamics(inner_size, action_space_size)
    prediction = Prediction(inner_size, action_space_size)

    num_iterations = 3000
    num_episodes = 10
    episodes = get_episodes(10, 1000, 1000, num_initial_states, num_iterations, representation, dynamics, prediction)

    assert len(episodes) == num_episodes
    assert len(episodes[0].states) == num_iterations
