import slider
import policy
import time
import torch
import memory
import random
import math
import torch.nn.functional as F
import nstep
import numpy as np

STATE_SPACE_SIZE = 8
ACTION_SPACE_SIZE = 4

policy_network = policy.PolicyNetwork(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)

goal_network = policy.GoalNetwork(STATE_SPACE_SIZE)

value_network = policy.ValueNetwork(STATE_SPACE_SIZE)
lagged_value_network = policy.ValueNetwork(STATE_SPACE_SIZE)
lagged_value_network.copy_weights(value_network)

value_network2 = policy.ValueNetwork(STATE_SPACE_SIZE)
lagged_value_network2 = policy.ValueNetwork(STATE_SPACE_SIZE)
lagged_value_network2.copy_weights(value_network2)

replay_memory_size = 100000
replay_memory = memory.ReplayMemory(replay_memory_size)

# export OMP_NUM_THREADS=1


def live(iterations: int, batch_size: int, improve_flag: bool, num_steps: int, render: bool, polyak: float):
    n_step = nstep.Nstep(num_steps)
    g = slider.Game()
    state = g.get_state()
    total_reward = 0
    start = time.time()
    for i in range(iterations):

        noise = np.random.normal(0, 0.05, STATE_SPACE_SIZE)
        goal = np.clip(goal_network.get_goal(state) + noise, -1, 1)
        action = policy_network.get_action(state, goal)

        if render:
            with torch.no_grad():
                value = value_network.forward(torch.from_numpy(state).unsqueeze(0).float(), torch.from_numpy(goal).unsqueeze(0).float())
                g.render(value.item(), state, 0, goal)

        reward, next_state = g.step(action)

        n_step.push(state, action, reward, goal)

        state = next_state

        if i >= num_steps:
            nstate, naction, nnext_state, nreward, goal, next_goal = n_step.get()
            replay_memory.push(nstate, naction, nnext_state, nreward, goal, next_goal)

        if improve_flag:
            improve_policy(batch_size*4)
            improve_value(batch_size, num_steps, polyak)
            if i % 2 == 0:
                improve_goal(batch_size)

        total_reward += reward

    end = time.time()
    print("Elapsed time: ", end-start)
    print(total_reward)


def init_memory(iters=replay_memory_size, num_steps=10):
    live(iters, 32, False, num_steps, False, 1)


def live_loop(lives, iterations, batch_size, num_steps=10, render=False, polyak=0.005):
    for i in range(lives):
        print(f'Iteration: {i} of {lives}')
        live(iterations, batch_size, True, num_steps, render, polyak)

def improve_value(batch_size, num_steps, polyak):
    batch = replay_memory.sample(batch_size)

    states = torch.FloatTensor([t.state for t in batch])
    next_states = torch.FloatTensor([t.next_state for t in batch])
    rewards = torch.FloatTensor([t.reward for t in batch])
    goals = torch.FloatTensor([t.goal for t in batch])
    next_goals = torch.FloatTensor([t.next_goal for t in batch])

    '''
        IMPROVE VALUE NETWORK
    '''
    believed_values = value_network.forward(states, goals)
    believed_values2 = value_network2.forward(states, goals)

    with torch.no_grad():
        next_values = lagged_value_network.forward(next_states, next_goals)
        next_values2 = lagged_value_network2.forward(next_states, next_goals)
        min_next_values = torch.min(next_values, next_values2)
        target_values = rewards.view(-1, 1) + math.pow(0.99, num_steps)*min_next_values

    value_loss = F.smooth_l1_loss(believed_values, target_values)
    update(value_network, value_loss)
    lagged_value_network.soft_update(value_network, polyak)

    value_loss2 = F.smooth_l1_loss(believed_values2, target_values)
    update(value_network2, value_loss2)
    lagged_value_network2.soft_update(value_network2, polyak)

def improve_policy(batch_size):
    batch = replay_memory.sample(batch_size)

    states = torch.FloatTensor([t.state for t in batch])
    actions = torch.LongTensor([t.action for t in batch])
    next_states = torch.FloatTensor([t.next_state for t in batch])

    policy_logits = policy_network.forward(states, next_states)
    policy_loss_function = torch.nn.CrossEntropyLoss()
    policy_loss = policy_loss_function(policy_logits, actions)
    update(policy_network, policy_loss)

def improve_goal(batch_size):
    batch = replay_memory.sample(batch_size)
    states = torch.FloatTensor([t.state for t in batch])
    goal_loss = -value_network.forward(states, goal_network.forward(states)).mean()
    # goals = goal_network.forward(states)
    # values1 = value_network.forward(states, goals)
    # values2 = value_network2.forward(states, goals)
    # min_values = torch.min(values1, values2)
    # goal_loss = -min_values.mean()
    update(goal_network, goal_loss)


def update(network, loss):
    network.opt.zero_grad()
    loss.backward()
    for param in network.parameters():
        param.grad.data.clamp_(-1, 1)
    network.opt.step()

def agent_loop(iterations, cd):
    with torch.no_grad():
        g = slider.Game()
        g.render(0, 0, 0, 0)
        for i in range(iterations):
            state = g.get_state()
            goal = goal_network.get_goal(state)
            value = value_network.forward(torch.from_numpy(state).unsqueeze(0).float(), torch.from_numpy(goal).unsqueeze(0).float())
            action = policy_network.get_action(state, goal)
            _, _ = g.step(action)
            #g.render(round(vs.data[0][0].item(), 2), round(vs.data[0][1].item(), 2), round(
             #   vs.data[0][2].item(), 2), round(vs.data[0][3].item(), 2))
            g.render(value.item(), state, 0, goal)
            time.sleep(cd)
