import numpy as np
import policy
import torch
import time
import copy
from multiprocessing import Pool, cpu_count
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from datetime import datetime

import slider
#import slider_jumper
import graphics

from policy import *
from MCTS import *
from get_data import *
from reanalyze import *
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

representation = Representation(
    num_initial_states, state_space_size, inner_size)
representation_optimizer = torch.optim.AdamW(
    representation.parameters(), lr=3e-4, weight_decay=1e-4)

dynamics = Dynamics(inner_size, action_space_size)
dynamics_optimizer = torch.optim.AdamW(
    dynamics.parameters(), lr=3e-4, weight_decay=1e-4)

prediction = Prediction(inner_size, action_space_size)
prediction_optimizer = torch.optim.AdamW(
    prediction.parameters(), lr=3e-4, weight_decay=1e-4)

projector = Projector(inner_size)
projector_optimizer = torch.optim.AdamW(
    projector.parameters(), lr=3e-4, weight_decay=1e-4)

simpredictor = Projector(inner_size)
simpredictor_optimizer = torch.optim.AdamW(
    simpredictor.parameters(), lr=3e-4, weight_decay=1e-4)

#replay_buffer = Replay_buffer(100)
replay_buffer = load_from_file('trajectories/trajs20230628')

def save(name):
    save_to_file(replay_buffer, f'trajectories/{name}')
    save_params(name)

def load(name):
    global replay_buffer
    replay_buffer = load_from_file(f'trajectories/{name}')
    load_weights(name)

def save_params(name):
    state = {
        "repr": representation.state_dict(),
        "repr_opt": representation_optimizer.state_dict(),
        "dyn": dynamics.state_dict(),
        "dyn_opt": dynamics_optimizer.state_dict(),
        "pred": prediction.state_dict(),
        "pred_opt": prediction_optimizer.state_dict(),
        "projector": projector.state_dict(),
        "projector_optimizer": projector_optimizer.state_dict(),
        "simpredictor": simpredictor.state_dict(),
        "simpredictor_optimizer": simpredictor_optimizer.state_dict(),
    }
    torch.save(state, f'./weights/{name}')


def load_weights(name):
    state = torch.load(f'./weights/{name}')

    representation.load_state_dict(state['repr'])
    representation_optimizer.load_state_dict(state['repr_opt'])
    dynamics.load_state_dict(state['dyn'])
    dynamics_optimizer.load_state_dict(state['dyn_opt'])
    prediction.load_state_dict(state['pred'])
    prediction_optimizer.load_state_dict(state['pred_opt'])
    projector.load_state_dict(state['projector'])
    projector_optimizer.load_state_dict(state['projector_optimizer'])
    simpredictor.load_state_dict(state['simpredictor'])
    simpredictor_optimizer.load_state_dict(state['simpredictor_optimizer'])


def get_data(n_episodes: int, max_episode_length: int, temperature: float, discount: float = 0.99):
    episodes = get_episodes(n_episodes, num_initial_states, max_episode_length,
                            representation, dynamics, prediction, temperature, gameFactory, discount)

    for e in episodes:
        logger.rewards.append(e.get_reward_sum())

    for e in episodes:
        replay_buffer.add_episode(e)


def do_reanalyze_episodes(n_episodes: int, num_unroll_steps:int, discount: float = 0.99):
    inds, episodes = replay_buffer.sample_episodes(n_episodes)
    reanalyzed_episodes = reanalze_episodes(
        episodes, num_initial_states, representation, dynamics, prediction, discount, action_space_size, num_unroll_steps)

    for (i, e) in enumerate(reanalyzed_episodes):
        replay_buffer.replace_episode(inds[i], e)


def train(batch_size: int = 1024, num_unroll_steps: int = 5):
    batch = replay_buffer.sample_batch(
        batch_size, num_initial_states, num_unroll_steps)

    train_on_batch(batch, representation, dynamics, prediction, representation_optimizer,
                   dynamics_optimizer, prediction_optimizer, projector, projector_optimizer, simpredictor, simpredictor_optimizer, game.action_space_size, num_unroll_steps, num_initial_states, logger)


#main(20, 6, 4000, 1000, 1024, 5, 0.1, 0.99)
def main(
    n_iters: int,
    n_episodes: int,
    max_episode_length: int,
    n_batches: int = 1000,
    batch_size: int = 1024,
    num_unroll_steps=5,
    temperature=1,
    discount=0.99
):
    for i in range(n_iters):
        print(f'Iteration {i} of {n_iters}. {datetime.now()}.')

        print("Gathering data...")
        get_data(n_episodes, max_episode_length, temperature, discount)
        print(logger.get_mean_rewards_of_last_n(10))
        main_counter.increment()

        print("Training...")
        for _ in range(n_batches):
            batch = replay_buffer.sample_batch(batch_size, num_initial_states, num_unroll_steps)
            train_on_batch(
                batch,
                representation,
                dynamics,
                prediction,
                representation_optimizer,
                dynamics_optimizer,
                prediction_optimizer,
                projector,
                projector_optimizer,
                simpredictor,
                simpredictor_optimizer,
                game.action_space_size,
                num_unroll_steps,
                num_initial_states,
                logger
            )
        #print("reanalyzing...")
        #do_reanalyze_episodes(2, num_unroll_steps)


'''

    WATCH GAME

'''


def clear(win):
    for item in win.items[:]:
        item.undraw()
    win.update()


def agent_loop(iterations: int, temperature: float = 1, num_simulations=50, discount=0.99):
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

        root = MCTS(initial_states, representation, dynamics,
                    prediction, action_space_size, num_simulations, discount)

        action = sample_action(root, temperature)
        #action = get_best_action(root)
        reward, _ = game.step(action)

        clear(win)
        game.render(win)
        graphics.Text(graphics.Point(300, 300), reward).draw(win)
        graphics.Text(graphics.Point(500, 500), "value:" +
                      str(root.search_value())).draw(win)
        graphics.Text(graphics.Point(500, 600), "rewards:" +
                      str(root.get_mean_reward())).draw(win)
    win.close()


def human_play(iterations: int, record: bool = False):
    win = graphics.GraphWin("canvas", game_width, game_height)
    win.setBackground('lightskyblue')

    game = gameFactory()

    key_to_action_map = {
        "Up": 1,
        "Right": 2,
        "Down": 3,
        "Left": 4,
        "space": 5
    }

    episode = Episode(game.state_space_size)

    for i in range(iterations):
        state = game.get_state()
        key = win.checkKey()

        action = key_to_action_map[key] if key in key_to_action_map else 0
        reward, _ = game.step(action)

        clear(win)
        game.render(win)
        graphics.Text(graphics.Point(300, 300), reward).draw(win)
        graphics.Text(graphics.Point(100, 100), "KEY:" + key).draw(win)

        action_one_hot = [0.8 if i == action else (
            0.2/(game.action_space_size-1)) for i in range(game.action_space_size)]

        episode.add_transition(reward, action, state, action_one_hot, 0.0)
        time.sleep(0.02)

    if record:
        episode.calc_targets(0.99)
        replay_buffer.add_episode(episode)
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
    #assert False


def test_MCTS():
    game = gameFactory()

    initial_states = [np.zeros(state_space_size)
                      for i in range(num_initial_states)]

    root = None
    for i in range(40):
        state = game.get_state()

        initial_states.pop(0)
        initial_states.append(state)

        root = MCTS(initial_states, representation, dynamics,
                    prediction, action_space_size, 50, .99)

        action = sample_action(root, 1)
        #action = get_best_action(root)
        reward, _ = game.step(action)

    print_tree(root, 0, 2)
    win = graphics.GraphWin("canvas", game_width, game_height)
    win.setBackground('lightskyblue')
    game.render(win)
    graphics.Text(graphics.Point(300, 300), reward).draw(win)
    graphics.Text(graphics.Point(500, 500), "value:" +
                  str(root.search_value())).draw(win)

    graphics.Text(graphics.Point(500, 600), "rewards:" +
                  str(root.get_mean_reward())).draw(win)
    # time.sleep(1)a
    assert True


def test_correct_actions_in_train_batch():
    initial_states = 7
    num_unroll_steps = 5
    main(2, 6, 100, 4, 8)
    batch = replay_buffer.sample_batch(3, initial_states, num_unroll_steps)
    targets = [e[1] for e in batch]
    g = gameFactory()
    one_hot_actions, search_policies, value_targets, observed_rewards, isNotDone = prepare_targets(
        targets, num_unroll_steps, g.action_space_size)

    actions = [[ep_targs[2] for ep_targs in e] for e in targets]

    print("------------------------")
    print(targets)
    print("------------------------")
    print(actions)
    print("------------------------")
    print(one_hot_actions)

    assert True
