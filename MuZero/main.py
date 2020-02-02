import numpy as np
import policy
import torch
import time
import copy
from multiprocessing import Pool, cpu_count
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import slider
import graphics

from policy import *
from MCTS import *
from get_data import *
from episode import *
from replay_buffer import *
from train import *
from logger import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"

ncpus = cpu_count()
game_width = 800
game_height = 700


def gameFactory(): return slider.Game(game_width, game_height)


logger = Logger()
game = gameFactory()

num_initial_states = 7
state_space_size = game.state_space_size
inner_size = 128
action_space_size = game.action_space_size

representation = Representation(num_initial_states, state_space_size, inner_size)
representation_optimizer = torch.optim.Adam(representation.parameters(), lr=3e-4)

dynamics = Dynamics(inner_size, action_space_size)
dynamics_optimizer = torch.optim.Adam(dynamics.parameters(), lr=3e-4)

prediction = Prediction(inner_size, action_space_size)
prediction_optimizer = torch.optim.Adam(prediction.parameters(), lr=3e-4)

replay_buffer = Replay_buffer(100)


def get_data(n_episodes: int, max_episode_length: int, temperature: float, discount: float = 0.99):
    episodes = get_episodes(n_episodes, num_initial_states, max_episode_length,
                            representation, dynamics, prediction, temperature, gameFactory)

    for e in episodes:
        logger.rewards.append(sum(e.rewards))

    for e in episodes:
        replay_buffer.add_episode(e, discount)


def train(batch_size: int = 1024, num_unroll_steps: int = 5, discount: float = .99):

    batch = replay_buffer.sample_batch(batch_size, num_initial_states, num_unroll_steps, discount)

    train_on_batch(batch, representation, dynamics, prediction, representation_optimizer,
                   dynamics_optimizer, prediction_optimizer, game.action_space_size, logger)


def main(n_iters: int, n_episodes: int, max_episode_length: int, n_batches: int = 1000, batch_size: int = 1024, temperature=1):
    for i in range(n_iters):
        print("gathering data...")
        get_data(n_episodes, max_episode_length, temperature)
        print(logger.get_mean_rewards_of_last_n(10))
        print("training...")
        for _ in range(n_batches):
            train(batch_size)


'''

    WATCH GAME

'''


def clear(win):
    for item in win.items[:]:
        item.undraw()
    win.update()


def agent_loop(iterations: int, temperature: float = 1):
    win = graphics.GraphWin("canvas", game_width, game_height)
    win.setBackground('lightskyblue')

    game = gameFactory()

    initial_states = [np.zeros(state_space_size)
                      for i in range(num_initial_states)]

    import time
    for i in range(iterations):
        state = game.get_state()

        initial_states.pop(0)
        initial_states.append(state)

        root = MCTS(initial_states, representation, dynamics, prediction, action_space_size, 50, .99)

        action = sample_action(root, temperature)
        #action = get_best_action(root)
        reward, _ = game.step(action)

        clear(win)
        game.render(win)
        graphics.Text(graphics.Point(300, 300), reward).draw(win)
        graphics.Text(graphics.Point(500, 500), "value:" + str(root.search_value())).draw(win)
        graphics.Text(graphics.Point(500, 600), "rewards:" +
                      str(root.get_mean_reward())).draw(win)
    win.close()


'''

    TEST get_data

'''


def test_get_data():
    get_data(20, 50, 1)
    assert len(replay_buffer.replay_buffer) == 20


'''
 random tests
'''


def test_main():
    main(2, 6, 100, 4, 8)
    print(logger.head_losses[-40:])
    assert False


def test_MCTS():
    game = gameFactory()

    initial_states = [np.zeros(state_space_size)
                      for i in range(num_initial_states)]

    root = None
    for i in range(40):
        state = game.get_state()

        initial_states.pop(0)
        initial_states.append(state)

        root = MCTS(initial_states, representation, dynamics, prediction, action_space_size, 50, .99)

        action = sample_action(root, 1)
        #action = get_best_action(root)
        reward, _ = game.step(action)

    print_tree(root, 0, 2)
    win = graphics.GraphWin("canvas", game_width, game_height)
    win.setBackground('lightskyblue')
    game.render(win)
    graphics.Text(graphics.Point(300, 300), reward).draw(win)
    graphics.Text(graphics.Point(500, 500), "value:" + str(root.search_value())).draw(win)

    graphics.Text(graphics.Point(500, 600), "rewards:" +
                  str(root.get_mean_reward())).draw(win)
    # time.sleep(1)a
    assert True




def test_correct_actions_in_train_batch():
    main(2, 6, 100, 4, 8)
    batch = replay_buffer.sample_batch(3, 7, 5, .99)
    targets = [e[1] for e in batch]
    num_unroll_steps = len(targets[0])
    one_hot_actions, search_policies, value_targets, observed_rewards, isNotDone = prepare_targets(
        targets, 4)

    actions = [[ep_targs[2] for ep_targs in e]for e in targets]

    print("------------------------")
    print(targets)
    print("------------------------")
    print(actions)
    print("------------------------")
    print(one_hot_actions)

    assert True
