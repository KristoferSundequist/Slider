from graphics import *
import numpy as np
import policy
import torch
import time
from multiprocessing import Pool, cpu_count
from slider import *
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

ncpus = cpu_count()
width = 700
height = 700
agent = policy.policy(Game.state_space_size, Game.action_space_size)
optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    
def getEpisode(n,agent):
    game = Game(width,height)
    
    states = np.zeros((n, game.state_space_size))
    actionprobs = []
    actions = np.zeros((n,game.action_space_size))
    values = []
    rewards = np.zeros(n)
    hiddens = []
    
    current_state = game.get_state()
    hidden = agent.init_hidden()
    
    for i in range(n):
        hiddens.append(hidden)
        actionprob, action, value, hidden = agent.get_action(current_state, hidden)

        states[i] = current_state
        actions[i] = np.eye(game.action_space_size)[action]
        actionprobs.append(actionprob)
        values.append(value)
        
        reward, current_state = game.step(action)

        rewards[i] = reward

    actionprobs = torch.cat(actionprobs).float().data
    values = torch.cat(values).view(-1).float().data
    hiddens = torch.cat(hiddens).float().data
    return states, actionprobs, actions, values, rewards, hiddens

#torch.save(policy.agent.state_dict(), PATH)
#policy.agent.load_state_dict(torch.load(PATH))

torch.multiprocessing.set_sharing_strategy('file_system')
running_reward = None

# export OMP_NUM_THREADS=1
#train(20,6000,20,0.03,3,0.1,0.99,0.95,0.0003,20, 128)
def train(num_actors, episode_size, episodes, beta, ppo_epochs, eps, gamma, lambd, lr, batch_size, rec_length):
    assert num_actors*episode_size > batch_size*rec_length, "num_actors*episode_size > batch_size*rec_length"
    time_start = time.time()
    global running_reward
    decay = 0.9
    
    torch.set_num_threads(1)
    pool = Pool(ncpus)
    torch.set_num_threads(ncpus)
    
    for i in range(episodes):
        tact = time.time()
        
        data = pool.starmap(getEpisode, [(episode_size,agent) for _ in range(num_actors)])
        
        data = list(zip(*data))
        states_data = list(data[0])
        actionprobs_data = list(data[1])
        actions_data = list(data[2])
        values_data = list(data[3])
        rewards = list(data[4])
        hiddens = list(data[5])
        advantage_data = [generalized_advantage_estimation(rewards[i],values_data[i].numpy(),gamma,lambd,values_data[i][-1]) \
                          for i in range(num_actors)]

        episode_reward_sums = np.array(rewards).sum(1)
        meanreward = episode_reward_sums.mean()
        if running_reward == None:
            running_reward = meanreward
        else:
            running_reward = decay*running_reward + (1-decay)*meanreward
            
        
        acc_states = np.concatenate(states_data)
        acc_actionprobs = torch.cat(actionprobs_data)
        acc_actions = np.concatenate(actions_data)
        acc_values = torch.cat(values_data)
        acc_advantages = np.concatenate(advantage_data)
        acc_hiddens = torch.cat(hiddens)
        
        print("actTime: ", time.time()-tact)
        ttrain = time.time()
        a,v,e = ppo(agent, acc_states, acc_actionprobs, acc_actions, acc_values, acc_advantages, acc_hiddens, \
            beta, ppo_epochs, eps, lr, batch_size, rec_length, 0.5)
        print("trainTime: ", time.time()-ttrain)
        #print(i+1, "/", episodes, running_reward, " ", meanreward, "Losses (action, value, entropy): ", a, v, e)
        print("Episode: ", i+1, "/", episodes, " Rewards: ", running_reward, " ", meanreward)
    print("Time: ", time.time()-time_start)
    pool.close()
    pool.join()

def ppo(agent,states,old_actionprobs,actions,values,advantages,hiddens,beta,ppo_epochs,eps,learning_rate,batch_size,rec_length,max_grad_norm=0.5):
    optimizer.learning_rate = learning_rate
    states = torch.from_numpy(states).float()
    actions = torch.from_numpy(actions).float()
    advantages = torch.from_numpy(advantages).float()
    old_actionprobs = old_actionprobs.float()
    
    returns = values+advantages
    normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-08)
    log_action_loss = log_value_loss = log_entropy_loss = 0

    for _ in range(ppo_epochs):
        sampler = BatchSampler(SubsetRandomSampler(range(0, len(states)-rec_length, rec_length)), batch_size,drop_last=True)
        for start_indices in sampler:
            start_indices = torch.LongTensor(start_indices)
            total_loss = 0

            for i in range(rec_length):
                indices = start_indices + i
                taken_actions = actions[indices]

                new_actionprobs, new_values, new_hiddens = agent.forward(states[indices], hiddens[indices].squeeze().unsqueeze(0))
                if i < rec_length -1:
                    hiddens[indices+1] = new_hiddens.squeeze().unsqueeze(1).detach()
                
                new_action = torch.sum(new_actionprobs*taken_actions,1)

                old_actionprobs_batch = old_actionprobs[indices]
                old_action_batch = torch.sum(old_actionprobs_batch*taken_actions,1)

                # Policy loss            
                entropy = -torch.sum(new_actionprobs * torch.log(new_actionprobs + 1e-08),1)
                ratio = new_action/old_action_batch
                adv = normalized_advantages[indices]
                surr1 = ratio*adv
                surr2 = torch.clamp(ratio, 1 - eps , 1 + eps)*adv
                action_loss = -torch.min(surr1,surr2).mean()

                # Value loss
                old_vals = values[indices]
                value_target = returns[indices]
                value_loss1 = (new_values.view(-1) - value_target).pow(2)
                clipped = old_vals + torch.clamp(new_values.view(-1) - old_vals, -eps, eps)
                value_loss2 = (clipped - value_target).pow(2)
                value_loss = torch.max(value_loss1, value_loss2).mean()

                loss = action_loss + .5*value_loss - beta*entropy.mean()
                total_loss += loss

            total_loss /= rec_length
            
            log_action_loss += action_loss
            log_value_loss += value_loss
            log_entropy_loss += entropy.mean()
            agent.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()
    num_updates = ppo_epochs + int(len(states)/batch_size)
    #return (log_action_loss/num_updates).item(),(log_value_loss/num_updates).item(),(log_entropy_loss/num_updates).item()
    return 0,0,0
    
def generalized_advantage_estimation(rewards,values,gamma,lambd,last):
    rewards = rewards.astype(float)
    values = np.append(values,last)
    gaes = np.zeros_like(rewards)
    gae = 0
    for i in reversed(range(len(gaes))):
        delta = rewards[i] + gamma*values[i+1] - values[i]
        gae = delta + gamma * lambd * gae
        gaes[i] = gae

    return gaes
        
def discount(r,gamma,init):
    dr = np.zeros_like(r)
    dr[dr.shape[0]-1] = init
    R = init
    for i in reversed(range(len(r))):
        R = r[i] + gamma*R
        dr[i] = R

    return dr

def clear(win):
    for item in win.items[:]:
        item.undraw()
    win.update()

def agent_loop(iterations):
    win = GraphWin("canvas", width, height)
    win.setBackground('lightskyblue')
    game = Game(width,height)
    hidden = agent.init_hidden()
    import time
    for i in range(iterations):
        _, action, v, hidden = agent.get_action(game.get_state(), hidden)
        _,_ = game.step(action)
            
        game.render(win)
        Text(Point(100,100), v.data[0][0]).draw(win)
        time.sleep(0.005)
        clear(win)
    win.close()
