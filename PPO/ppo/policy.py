import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class policy(nn.Module):
    def __init__(self):
        super(policy, self).__init__()

        self.ainp = nn.Linear(8,200)
        self.a0 = nn.Linear(200,200)
        self.a1 = nn.Linear(200,200)
        self.action_out = nn.Linear(200,4)

        self.vinp = nn.Linear(8,200)
        self.v0 = nn.Linear(200,200)
        self.v1 = nn.Linear(200,200)
        self.value_out = nn.Linear(200,1)
        
        self.apply(self.weight_init)
        self.opt = optim.Adam(self.parameters(), lr=1e-03, eps=1e-05)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            size = m.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            variance = np.sqrt(2.0/(fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)

    def save_weights(self, name):
        torch.save(self.state_dict(), name)

    def load_weights(self, name):
        self.load_state_dict(torch.load(name))
        
    def forward(self, state):
        
        a = F.relu(self.ainp(state))
        a = F.relu(self.a0(a))
        a = F.relu(self.a1(a))
        actions = F.softmax(self.action_out(a), dim=1)

        v = F.relu(self.vinp(state))        
        v = F.relu(self.v0(v))
        v = F.relu(self.v1(v))
        value = self.value_out(v)
        
        return actions, value

    def train(self,states,old_actionprobs,actions,values,returns,advantages,beta,ppo_epochs,eps,learning_rate,batch_size, use_critic):
    #def train(self,states,old_actionprobs,actions,values,returns,beta,ppo_epochs,eps,learning_rate,batch_size, use_critic):
        self.opt.learning_rate = learning_rate
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions)
        #returns = torch.from_numpy(returns).float()
        advantages = torch.from_numpy(advantages).float()
        returns = values+advantages

        #advantages = returns - values
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-08)
        #normalized_advantages = advantages / advantages.std()
        
#        normalized_returns = (returns - returns.mean()) / (returns.std() + 1e-08)

        log_action_loss = log_value_loss = log_entropy_loss = 0
    
        for _ in range(ppo_epochs):
            sampler = BatchSampler(SubsetRandomSampler(range(len(states))), batch_size,drop_last=False)
            for indices in sampler:
                indices = torch.LongTensor(indices)
                taken_actions = actions[indices].float()
                
                new_actionprobs, new_values = self.forward(states[indices])
                new_action = torch.sum(new_actionprobs*taken_actions,1)
                
                old_actionprobs_batch = old_actionprobs[indices].float()
                old_action_batch = torch.sum(old_actionprobs_batch*taken_actions,1)
                
                entropy = -torch.sum(new_actionprobs * torch.log(new_actionprobs + 1e-08),1)
                
                ratio = new_action/old_action_batch

                if use_critic:
                    adv = normalized_advantages[indices]
                else:
                    adv = normalized_returns[indices]
                    
                surr1 = ratio*adv
                surr2 = torch.clamp(ratio, 1 - eps , 1 + eps)*adv
                action_loss = -torch.min(surr1,surr2).mean()

                if use_critic:
                    old_vals = values[indices]
                    
                    value_target = returns[indices]
                    value_loss1 = (new_values.view(-1) - value_target).pow(2)
                
                    clipped = old_vals + torch.clamp(new_values.view(-1) - old_vals, -eps, eps)
                    value_loss2 = (clipped - value_target).pow(2)
                
                    value_loss = torch.max(value_loss1, value_loss2).mean()
                    
                    loss = action_loss + .5*value_loss - beta*entropy.mean()
                else:
                    value_loss = torch.tensor([0])
                    loss = action_loss - beta*entropy.mean()


                log_action_loss += action_loss
                log_value_loss += value_loss
                log_entropy_loss += entropy.mean()

                self.zero_grad()
                loss.backward()
                self.opt.step()

        num_updates = ppo_epochs + int(len(states)/batch_size)
        return (log_action_loss/num_updates).item(),(log_value_loss/num_updates).item(),(log_entropy_loss/num_updates).item()

    def get_action(self, state):
        with torch.no_grad():
            out,value = self.forward(torch.from_numpy(state).view(1,8).float())
            return out, out.multinomial(1).item(), value




