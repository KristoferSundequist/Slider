import torch
from torch import autograd, nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class policy(nn.Module):
    def __init__(self):
        super(policy, self).__init__()

        self.shared = nn.Linear(8,200)
        self.seq = nn.GRU(200,200)
        
        self.a1 = nn.Linear(200,200)
        self.action_out = nn.Linear(200,4)

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

    def init_hidden(self):
        return torch.randn(1,1,200, requires_grad=False)
        
    def forward(self, state, hidden):

        x = F.selu(self.shared(state))
        x, hidden = self.seq(x.view(1,-1,200), hidden)
        x = x.view(-1,200)
        a1 = F.selu(self.a1(x))
        actions = F.softmax(self.action_out(a1), dim=1)

        v1 = F.selu(self.v1(x))
        value = self.value_out(v1)
        
        return actions, value, hidden

    def train(self,states,old_actionprobs,actions,values,returns,beta,ppo_epochs,eps,learning_rate,batch_size, use_critic, hiddens):
        self.opt.learning_rate = learning_rate
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()
        returns = torch.from_numpy(returns).float()

        advantages = returns - values
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-08)
        #normalized_advantages = advantages / advantages.std()
        
        normalized_returns = (returns - returns.mean()) / (returns.std() + 1e-08)
        
        rec_length = 10
        for _ in range(ppo_epochs):
            sampler = BatchSampler(SubsetRandomSampler(range(len(states)-rec_length)), batch_size,drop_last=False)
            for startinds in sampler:
                startinds = torch.LongTensor(startinds)
                total_loss = 0
                
                for i in range(rec_length):
                    indices = startinds + i
                    
                    taken_actions = actions[indices]

                    new_actionprobs, new_values, new_hiddens = self.forward(states[indices], hiddens[indices].squeeze().unsqueeze(0))
                    if i < rec_length -1:
                        hiddens[indices+1] = new_hiddens.squeeze().unsqueeze(1).detach()
                        
                    new_action = torch.sum(new_actionprobs*taken_actions.float(),1)
                
                    old_actionprobs_batch = old_actionprobs[indices]
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
                    
                        loss = action_loss + 0.5*value_loss - beta*entropy.mean()
                    else:
                        loss = action_loss - beta*entropy.mean()

                    total_loss += loss

                total_loss /= rec_length
                
                self.zero_grad()
                total_loss.backward()
                self.opt.step()

    def get_action(self, state, hidden):
        out,value,hidden = self.forward(torch.from_numpy(state).view(1,8).float(), hidden)
        return out, out.multinomial(1).item(), value, hidden




