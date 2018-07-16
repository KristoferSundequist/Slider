import torch
from torch import autograd, nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class policy(nn.Module):
    def __init__(self):
        super(policy, self).__init__()

        self.a0 = nn.Linear(8,200)
        self.a1 = nn.Linear(200,200)
        self.a2 = nn.Linear(200,200)
        self.action_out = nn.Linear(200,4)

        self.v0 = nn.Linear(8,200)
        self.v1 = nn.Linear(200,200)
        self.value_out = nn.Linear(200,1)
        
        self.apply(self.weight_init)
        self.opt = optim.Adam(self.parameters(), lr=0.0001)

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
        
        a0 = F.selu(self.a0(state))
        a1 = F.selu(self.a1(a0))
        a2 = F.selu(self.a2(a1))
        actions = F.softmax(self.action_out(a2), dim=1)

        v0 = F.selu(self.v0(state))
        v1 = F.selu(self.v1(v0))
        value = self.value_out(v1)
        
        return actions, value

    def train(self,states,old_actionprobs,actions,values,returns,beta,ppo_epochs,eps,learning_rate,batch_size, use_critic):
        self.opt.learning_rate = learning_rate
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions)
        returns = torch.from_numpy(returns).float()

        advantages = returns - values
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-08)
        #normalized_advantages = advantages / advantages.std()
        
        normalized_returns = (returns - returns.mean()) / (returns.std() + 1e-08)
        
    
        for _ in range(ppo_epochs):
            sampler = BatchSampler(SubsetRandomSampler(range(len(states))), batch_size,drop_last=False)
            for indices in sampler:
                indices = torch.LongTensor(indices)
                taken_actions = Variable(actions[indices]).float()
                
                new_actionprobs, new_values = self.forward(Variable(states[indices]))
                new_action = torch.sum(new_actionprobs*taken_actions,1)
                
                old_actionprobs_batch = Variable(old_actionprobs[indices]).float()
                old_action_batch = torch.sum(old_actionprobs_batch*taken_actions,1)
                
                entropy = -torch.sum(new_actionprobs * torch.log(new_actionprobs + 1e-08),1)
                
                ratio = new_action/old_action_batch

x                if use_critic:
                    adv = Variable(normalized_advantages[indices])
                else:
                    adv = Variable(normalized_returns[indices])
                    
                surr1 = ratio*adv
                surr2 = torch.clamp(ratio, 1 - eps , 1 + eps)*adv
                action_loss = -torch.min(surr1,surr2).mean()

                if use_critic:
                    old_vals = Variable(values[indices])
                    
                    value_target = Variable(returns[indices])
                    value_loss1 = (new_values.view(-1) - value_target).pow(2)
                
                    clipped = old_vals + torch.clamp(new_values.view(-1) - old_vals, -eps, eps)
                    value_loss2 = (clipped - value_target).pow(2)
                
                    value_loss = torch.max(value_loss1, value_loss2).mean()
                    
                    loss = action_loss + 0.5*value_loss - beta*entropy.mean()
                else:
                    loss = action_loss - beta*entropy.mean()

                self.zero_grad()
                loss.backward()
                self.opt.step()

    def get_action(self, state):    
        out,value = self.forward(Variable(torch.from_numpy(state), volatile=True).view(1,8).float())
        return out, out.multinomial().data[0][0], value




