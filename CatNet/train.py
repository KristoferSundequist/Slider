import slider
#import simple_slider
import time
import torch
from graphics import *
import memory
import random
import math
import torch.nn.functional as F
import nstep
import globals
import numpy as np
import agent
from transition import *
from logger import *


# game = simple_slider.Game
game = slider.Game

replay_memory = memory.ReplayMemory(globals.replay_memory_size)

# export OMP_NUM_THREADS=1


agent = agent.Agent(game.state_space_size, game.action_space_size)
logger = Logger()


def live(iterations, improve_flag):
    n_step = nstep.Nstep()
    g = game()
    state = g.get_state()
    total_episode_reward = 0
    start = time.time()
    for i in range(iterations):

        # eps-greedy
        if random.uniform(0, 1) < globals.epsilon:
            action = random.randint(0, game.action_space_size-1)
        else:
            action = agent.get_action(state)

        reward, next_state = g.step(action)

        n_step.push(state, action, reward)

        state = next_state

        if i > globals.nsteps:
            nstate, naction, nnext_state, nreward = n_step.get()
            replay_memory.push(
                Transition(nstate, naction, nnext_state, nreward)
            )

        if len(replay_memory) > 100 and improve_flag:
            for _ in range(globals.updates_per_step):
                agent.improve(replay_memory.sample(globals.batch_size))

        total_episode_reward += reward

    logger.add_reward(total_episode_reward)
    print(f'avg_running_reward: {logger.get_running_avg_reward()}, total_episode_reward: {total_episode_reward}')

    end = time.time()
    print("Elapsed time: ", end-start)


def live_loop(lives, iterations, should_reset):
    for i in range(lives):
        print(f'Iteration: {i} of {lives}')
        live(iterations, True)

        if should_reset and i != 0 and i % globals.reset_freq == 0:
            print("Resetting models")
            start = time.time()
            agent.reset(replay_memory)
            end = time.time()
            print("Resetting complete. Elapsed time: ", end-start)


def agent_loop(iterations, cd=0.01):
    win = GraphWin("canvas", globals.width, globals.height)
    win.setBackground('lightskyblue')
    with torch.no_grad():
        g = game()
        g.render(0, 0, 0, 0, win)
        for i in range(iterations):
            parsed_state = torch.from_numpy(g.get_state()).view(1, -1).float().to(globals.device)
            vs = agent.get_expected_values(parsed_state)
            action = torch.max(vs, 1)[1].data[0]
            _, _ = g.step(action)
            g.render(
                round(vs.data[0][0].item(), 2),
                round(vs.data[0][1].item(), 2),
                round(vs.data[0][2].item(), 2),
                round(vs.data[0][3].item(), 2),
                win
            )
            # g.render(0,0,0,0)
            time.sleep(cd)
    win.close()
