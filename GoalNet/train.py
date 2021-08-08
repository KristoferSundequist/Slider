import slider
import policy
import time
import torch
import memory
import random
import math
import torch.nn.functional as F
import nstep

agent = policy.policy()

lagged_agent = policy.policy()
lagged_agent.copy_weights(agent)

replay_memory_size = 100000
replay_memory = memory.ReplayMemory(replay_memory_size)

# export OMP_NUM_THREADS=1
def live(iterations, batch_size, lagg, eps, improve_flag, num_steps):
    n_step = nstep.Nstep(num_steps)
    g = slider.Game()
    state = g.get_state()
    total_reward = 0
    start = time.time()
    for i in range(iterations):

        # eps-greedy
        if random.uniform(0,1) < eps:
            action = random.randint(0,3)
        else:
            action = agent.get_action(state)
            
        reward,next_state = g.step(action)
        
        n_step.push(state, action, reward)
        
        state = next_state
        
        if i >= num_steps:
            nstate, naction, nnext_state, nreward = n_step.get()
            replay_memory.push(nstate, naction, nnext_state, nreward)
            
        if improve_flag:
            improve(batch_size, num_steps)

        if i % lagg == 0:
            lagged_agent.copy_weights(agent)

        total_reward += reward

    end = time.time()
    print("Elapsed time: ", end-start)
    print(total_reward)

def init_memory(iters=replay_memory_size, eps=1, num_steps=3):
    live(iters, 32, 999999, eps, False, num_steps)
    
def live_loop(lives, iterations, batch_size, lagg, eps, num_steps=3):
    for i in range(lives):
        print(f'Iteration: {i} of {lives}')
        live(iterations, batch_size, lagg, eps, True, num_steps)

def improve(batch_size, num_steps):
    batch = replay_memory.sample(batch_size)
    
    states = torch.FloatTensor([t.state for t in batch])
    actions = torch.LongTensor([t.action for t in batch])
    next_states = torch.FloatTensor([t.next_state for t in batch])
    rewards = torch.FloatTensor([t.reward for t in batch])

    believed_qvs = agent.forward(states).gather(1,actions.view(-1,1))

    # double dqn
    with torch.no_grad():
        next_actions = agent.forward(next_states).max(1)[1].view(-1,1)
        next_qvs = lagged_agent.forward(next_states).gather(1, next_actions)
        target_v = rewards.view(-1,1) + math.pow(0.99,num_steps)*next_qvs;

    loss = F.smooth_l1_loss(believed_qvs, target_v)

    agent.opt.zero_grad()
    loss.backward()
    for param in agent.parameters():
        param.grad.data.clamp_(-1,1)
    agent.opt.step()

def agent_loop(iterations, cd):
    with torch.no_grad():
        g = slider.Game()
        g.render(0,0,0,0)
        for i in range(iterations):
            state = g.get_state()
            vs = agent.forward(torch.from_numpy(state).view(1,8).float())
            action = torch.max(vs,1)[1].data[0]
            _,_ = g.step(action)        
            g.render(round(vs.data[0][0].item(), 2), round(vs.data[0][1].item(), 2), round(vs.data[0][2].item(), 2), round(vs.data[0][3].item(), 2))
            #g.render(0,0,0,0)
            time.sleep(cd)
