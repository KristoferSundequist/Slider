import graphics
import numpy as np
import time
import copy
from multiprocessing import Pool, cpu_count
from slider import *
import random
from utils import calculate_value_targets, arrange_data, TrainingData
import globals
from typing import *
from episode import Episode
from agent import Agent

ncpus = cpu_count()
width = 800
height = 700

game = Game

agent = Agent(game.state_space_size, game.action_space_size)


def get_episode(num_iterations: int, epsilon: float):
    env = game(width, height)

    states = []
    actions = []
    values = []
    rewards = []

    current_state = env.get_state()

    for i in range(num_iterations):
        qvalues = agent.get_qvalues(current_state)
        if random.random() <= epsilon:
            action = random.randint(0, Game.action_space_size - 1)
        else:
            action = qvalues.argmax()

        reward, next_state = env.step(action)

        states.append(current_state)
        actions.append(action)
        values.append(qvalues)
        rewards.append(reward)

        current_state = next_state

    return Episode(
        states,
        actions,
        values,
        rewards,
        calculate_value_targets(rewards, actions, values, globals.discount_factor, globals.lambd_factor),
        sum(rewards),
    )


running_reward = None
running_reward_decay = 0.9


def train(
    n_episodes: int,
    episode_size: int,
    epsilon: float,
):
    global running_reward

    for i in range(n_episodes):
        print(f"Episode {i+1} / {n_episodes}.")
        act_time_start = time.time()
        episodes = [get_episode(episode_size, epsilon) for _ in range(20)]

        mean_reward = np.array([e.reward_sum for e in episodes]).mean()
        if running_reward == None:
            running_reward = mean_reward
        else:
            running_reward = running_reward_decay * running_reward + (1 - running_reward_decay) * mean_reward
        print(
            f"Act time elapsed: {time.time() - act_time_start}. Mean reward: {mean_reward}. Running avg: {running_reward}"
        )

        train_train_start = time.time()
        arranged_data = arrange_data(episodes, Game.action_space_size)

        agent.fit(arranged_data.states_and_actions, arranged_data.value_targets)

        print("Train time elapsed: ", time.time() - train_train_start)


def clear(win):
    for item in win.items[:]:
        item.undraw()
    win.update()


def agent_loop(iterations):
    win = graphics.GraphWin("canvas", width, height)
    win.setBackground("lightskyblue")

    env = game(width, height)
    import time

    for i in range(iterations):
        qvalues = agent.get_qvalues(env.get_state())
        action = int(qvalues.argmax())
        _, _ = env.step(action)

        env.render(win)
        value = str(qvalues[action])
        graphics.Text(graphics.Point(100, 100), value).draw(win)
        time.sleep(0.005)
        clear(win)
    win.close()
