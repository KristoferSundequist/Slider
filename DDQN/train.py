import slider
import policy
import time
import torch
import memory
from torch import autograd, nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import namedtuple


agent = policy.policy()

lagged_agent = policy.policy()
lagged_agent.copy_weights(agent)

replay_memory_size = 100000
replay_memory = memory.ReplayMemory(replay_memory_size)

# export OMP_NUM_THREADS=1
def live(iterations, batch_size, lagg, eps, improve_flag):
    g = slider.Game()
    state = g.get_state()
    total_reward = 0
    start = time.time()
    for i in range(iterations):
        action = agent.get_action(state, eps)
        reward,next_state = g.step(action)
        replay_memory.push(state, action, next_state, reward)
        state = next_state

        if improve_flag:
            improve(batch_size)

        if i % lagg == 0:
            lagged_agent.copy_weights(agent)

        total_reward += reward

    end = time.time()
    print("Elapsed time: ", end-start)
    print(total_reward)

def init_memory(iters=replay_memory_size, eps=1):
    live(iters, 32, 999999, eps, False)
    
def live_loop(lives, iterations, batch_size, lagg, eps):
    for _ in range(lives):
        live(iterations, batch_size, lagg, eps, True)

def improve(batch_size):
    #batch,indices = replay_memory.sample(batch_size, alpha)
    batch = replay_memory.sample(batch_size)
    
    states = Variable(torch.FloatTensor([t.state for t in batch]))
    actions = Variable(torch.LongTensor([t.action for t in batch]))
    next_states = Variable(torch.FloatTensor([t.next_state for t in batch]))
    rewards = Variable(torch.FloatTensor([t.reward for t in batch]))

    believed_qvs = agent.forward(states).gather(1,actions.view(-1,1))

    # double dqn
    next_actions = agent.forward(next_states).max(1)[1].view(-1,1)
    next_qvs = lagged_agent.forward(next_states).gather(1, next_actions)
    
    target_v = rewards.view(-1,1) + 0.99*next_qvs;
    target_v = Variable(target_v.data)
    
    # Update td errors in prioritized replay buffer 
    #td_errors = np.abs((target_v.data - believed_qvs.data).view(-1).numpy())
    #replay_memory.update_td_errors(indices, td_errors)

    #weights = torch.from_numpy(np.power(td_errors,-0.5)).view(-1,1)
    #weights = Variable((weights/torch.max(weights)).float())
    
    #loss = agent.weighted_loss(believed_qvs, target_v, weights)
    loss = F.smooth_l1_loss(believed_qvs, target_v)

    agent.opt.zero_grad()
    loss.backward()
    for param in agent.parameters():
        param.grad.data.clamp_(-1,1)
    agent.opt.step()

def agent_loop(iterations):
    g = slider.Game()
    for i in range(iterations):
        action = agent.get_action(g.get_state(), 0)
        _,_ = g.step(action)

        g.render()
        time.sleep(0.01)
