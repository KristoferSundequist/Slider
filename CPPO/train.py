import graphics
import numpy as np
from network import TheNetwork
import torch
import time
import copy
from multiprocessing import Pool, cpu_count
from slider import *
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Normal, Independent
import random
from utils import calculate_value_targets, arrange_data, TrainingData
import globals
from typing import *
from episode import Episode
import torch.nn.functional as F

ncpus = cpu_count()
width = 800
height = 700

agent = TheNetwork(Game.state_space_size, Game.action_space_size)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("DEVICE:", device)


def get_episode(num_iterations: int):
    with torch.no_grad():
        game = Game(width, height)

        all_states = []
        all_taken_actions = []
        all_action_means = []
        all_action_stds = []
        all_values = []
        all_rewards = []

        current_state = game.get_state()

        for i in range(num_iterations):
            current_values, action_means, action_stds = agent.forward(torch.tensor([current_state]))
            action = Normal(action_means, F.softplus(action_stds)).sample().tolist()[0]

            reward, next_state = game.step(action)

            all_states.append(current_state)
            all_taken_actions.append(action)
            all_action_means.append(action_means.tolist()[0])
            all_action_stds.append(action_stds.tolist()[0])
            all_values.append(current_values.item())
            all_rewards.append(reward)

            current_state = next_state

        return Episode(
            states=all_states,
            actions=all_taken_actions,
            values=all_values,
            rewards=all_rewards,
            action_means=all_action_means,
            action_stds=all_action_stds,
            value_targets=calculate_value_targets(
                all_rewards, all_values, globals.discount_factor, globals.lambd_factor
            ),
            reward_sum=sum(all_rewards),
        )


running_reward = None
running_reward_decay = 0.9


def train(
    n_iterations: int,
    n_episodes_per_iteration: int,
    episode_size: int,
):
    global running_reward

    for i in range(n_iterations):
        print(f"Episode {i+1} / {n_iterations}.")
        act_time_start = time.time()
        episodes = [get_episode(episode_size) for _ in range(n_episodes_per_iteration)]

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
    all_advantages = trainingData.value_targets - trainingData.values
    all_normalized_advantages = (all_advantages - all_advantages.mean()) / all_advantages.std()
    for _ in range(globals.pvo_epochs):
        sampler = BatchSampler(range(0, len(trainingData.states)), globals.batch_size, drop_last=True)
        for indices in sampler:
            new_values, new_action_means, new_action_stds = agent.forward(trainingData.states[indices])
            new_values = new_values.squeeze(1)

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
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

            # Action loss
            old_taken_actions = trainingData.actions[indices]
            old_action_means = trainingData.action_means[indices]
            old_action_stds = trainingData.action_stds[indices]
            old_dist = Independent(Normal(old_action_means, F.softplus(old_action_stds)), 1)
            new_dist = Independent(Normal(new_action_means, F.softplus(new_action_stds)), 1)
            advantages = all_normalized_advantages[indices]
            assert (
                new_action_means.size()
                == new_action_stds.size()
                == old_taken_actions.size()
                == old_action_means.size()
                == old_action_stds.size()
            ), f"{new_action_means.size()} == {new_action_stds.size()}  == {old_taken_actions.size()} == {old_action_means.size()} == {old_action_stds.size()}"

            new_log_prob = new_dist.log_prob(old_taken_actions)
            old_log_prob = old_dist.log_prob(old_taken_actions)
            ratio = torch.exp(new_log_prob - old_log_prob)
            assert ratio.size() == advantages.size()
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - globals.update_clamp_threshold, 1 + globals.update_clamp_threshold) * advantages
            )
            action_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -globals.entropy_coef * new_dist.entropy().mean()

            # Total_loss

            total_loss = value_loss + action_loss + entropy_loss

            agent.opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(agent.parameters(), globals.max_grad_norm)
            agent.opt.step()


def clear(win):
    for item in win.items[:]:
        item.undraw()
    win.update()


def agent_loop(iterations, std_coef: float):
    with torch.no_grad():
        win = graphics.GraphWin("canvas", width, height)
        win.setBackground("lightskyblue")

        game = Game(width, height)
        import time

        for i in range(iterations):
            values, action_means, action_stds = agent.forward(torch.tensor([game.get_state()]))
            action = Normal(action_means, std_coef * F.softplus(action_stds)).sample().tolist()[0]
            _, _ = game.step(action)

            game.render(win)
            value = str(values.item())
            graphics.Text(graphics.Point(100, 100), value).draw(win)
            time.sleep(0.005)
            clear(win)
        win.close()
