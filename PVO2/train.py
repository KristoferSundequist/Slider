import graphics
import numpy as np
from policy import QNetwork
import torch
import time
import copy
from multiprocessing import Pool, cpu_count
from slider import *
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import random
from utils import calculate_value_targets, arrange_data, TrainingData
import globals
from typing import *
from episode import Episode

ncpus = cpu_count()
width = 800
height = 700

agent = QNetwork(Game.state_space_size, Game.action_space_size)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("DEVICE:", device)


def get_episode(num_iterations: int, epsilon: float):
    with torch.no_grad():
        game = Game(width, height)

        states = []
        actions = []
        values = []
        rewards = []

        current_state = game.get_state()

        for i in range(num_iterations):
            qvalues = agent.forward(
                torch.tensor([current_state]).repeat(Game.action_space_size, 1), torch.eye(Game.action_space_size)
            )
            if random.random() <= epsilon:
                action = random.randint(0, Game.action_space_size - 1)
            else:
                action = int(qvalues.argmax().item())

            reward, next_state = game.step(action)

            states.append(current_state)
            actions.append(action)
            values.append(qvalues[action].item())
            rewards.append(reward)

            current_state = next_state

        return Episode(
            states,
            actions,
            values,
            rewards,
            calculate_value_targets(rewards, values, globals.discount_factor, globals.lambd_factor),
            sum(rewards),
        )


running_reward = None
running_reward_decay = 0.9


def train(
    n_iterations: int,
    n_epsiodes_per_iteration: int,
    episode_size: int,
    epsilon: float,
):
    global running_reward

    for i in range(n_iterations):
        print(f"Episode {i+1} / {n_iterations}.")
        act_time_start = time.time()
        episodes = [get_episode(episode_size, epsilon) for _ in range(n_epsiodes_per_iteration)]

        mean_reward = np.array([e.reward_sum for e in episodes]).mean()
        if running_reward == None:
            running_reward = mean_reward
        else:
            running_reward = running_reward_decay * running_reward + (1 - running_reward_decay) * mean_reward
        print(
            f"Act time elapsed: {time.time() - act_time_start}. Mean reward: {mean_reward}. Running avg: {running_reward}"
        )

        train_train_start = time.time()
        arranged_data = arrange_data(episodes, Game.state_space_size, Game.action_space_size)
        pvo(arranged_data)

        print("Train time elapsed: ", time.time() - train_train_start)


def pvo(trainingData: TrainingData):
    for _ in range(globals.pvo_epochs):
        sampler = BatchSampler(range(0, len(trainingData.states)), globals.batch_size, drop_last=True)
        for indices in sampler:
            new_values = agent.forward(trainingData.states[indices], trainingData.one_hot_actions[indices]).squeeze(1)

            # Value loss
            old_vals = trainingData.values[indices]
            value_target = trainingData.value_targets[indices]

            assert new_values.size() == value_target.size(), f"{new_values.size()} == {value_target.size()}"
            value_loss1 = (new_values - value_target).pow(2)

            assert old_vals.size() == new_values.size(), f"{old_vals.size()} == {new_values.size()}"
            clipped = old_vals + torch.clamp(
                new_values - old_vals, -globals.update_clamp_threshold, globals.update_clamp_threshold
            )

            assert clipped.size() == value_target.size()
            value_loss2 = (clipped - value_target).pow(2)
            value_loss = torch.max(value_loss1, value_loss2).mean()

            agent.opt.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(agent.parameters(), globals.max_grad_norm)
            agent.opt.step()


def clear(win):
    for item in win.items[:]:
        item.undraw()
    win.update()


def agent_loop(iterations):
    with torch.no_grad():
        win = graphics.GraphWin("canvas", width, height)
        win.setBackground("lightskyblue")

        game = Game(width, height)
        import time

        for i in range(iterations):
            values = agent.forward(
                torch.tensor([game.get_state()]).repeat(Game.action_space_size, 1), torch.eye(Game.action_space_size)
            )
            action = int(values.argmax().item())
            _, _ = game.step(action)

            game.render(win)
            value = str(values[action].item())
            graphics.Text(graphics.Point(100, 100), value).draw(win)
            time.sleep(0.005)
            clear(win)
        win.close()
