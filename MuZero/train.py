import numpy as np
import policy
import torch
import time
import copy
from multiprocessing import Pool, cpu_count
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from graphics import *
from slider import *
from policy import *
from MCTS import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"

ncpus = cpu_count()
width = 1500
height = 1000

num_initial_states = 7
state_space_size = Game.state_space_size
inner_size = 30
action_space_size = Game.action_space_size

representation = Representation(num_initial_states, state_space_size, inner_size)
dynamics = Dynamics(inner_size, action_space_size)
prediction = Prediction(inner_size, action_space_size)



'''

    WATCH GAME

'''

def clear(win):
    for item in win.items[:]:
        item.undraw()
    win.update()


def agent_loop(iterations):
    win = GraphWin("canvas", width, height)
    win.setBackground('lightskyblue')

    game = Game(width, height)

    initial_states = [np.zeros(state_space_size)
                      for i in range(num_initial_states)]


    import time
    for i in range(iterations):
        state = game.get_state()

        initial_states.pop(0)
        initial_states.append(state)
        
        root = MCTS(initial_states, representation, dynamics, prediction, action_space_size, 50, .99)

        action = sample_action(root)
        #action = get_best_action(root)
        _,_ = game.step(action)

        game.render(win)
        #Text(Point(300, 300), list(map(lambda v: round(v, 2), root.mean_values))).draw(win)

            #time.sleep(0.001)
        if i % 20 == 0:
            clear(win)
    win.close()
