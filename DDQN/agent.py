import policy
import globals
from typing import *
import torch
import numpy as np
import memory
from transition import *
import torch.nn.functional as F
import math
from sub_agent import *
import random


class Agent:
    def __init__(self, state_space_size: int, action_space_size: int):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

        self.sub_agent = Sub_Agent(state_space_size, action_space_size)
        self.sub_agent2 = Sub_Agent(state_space_size, action_space_size)

    def reset(self, replay_buffer: memory.ReplayMemory):
        self.sub_agent.reset()
        self.sub_agent2.reset()

        for i in range(globals.reset_retrain_iters):
            self.improve(replay_buffer.sample(globals.batch_size))
            if i % 2000 == 0:
                self.sub_agent.update_target()
                self.sub_agent2.update_target()

    def get_action(self, state: np.ndarray) -> int:
        agent_to_do_action = random.choice([0, 1])
        if agent_to_do_action == 0:
            return self.sub_agent.get_action(state)
        else:
            return self.sub_agent2.get_action(state)

    def get_values(self, state: np.ndarray) -> torch.tensor:
        return self.sub_agent.get_values(state)

    def improve(self, batch: List[Transition]):
        states = torch.FloatTensor(
            np.array([t.state for t in batch])).to(globals.device)
        actions = torch.LongTensor(
            [t.action for t in batch]).to(globals.device)
        next_states = torch.FloatTensor(
            np.array([t.next_state for t in batch])).to(globals.device)
        rewards = torch.FloatTensor(
            [t.reward for t in batch]).to(globals.device)

        believed_qvs = self.sub_agent.forward(
            states).gather(1, actions.view(-1, 1))
        believed_qvs2 = self.sub_agent2.forward(
            states).gather(1, actions.view(-1, 1))

        # double dqn
        with torch.no_grad():
            next_all_qvs = self.sub_agent.forward(next_states)

            next_actions = next_all_qvs.max(1)[1].view(-1, 1)

            next_qvs1 = next_all_qvs.gather(1, next_actions)
            next_qvs2 = self.sub_agent2.forward(
                next_states).gather(1, next_actions)

            next_qvs = torch.minimum(next_qvs1, next_qvs2)

            assert (next_qvs.size() == next_qvs1.size())
            target_v = rewards.view(-1, 1) + \
                math.pow(globals.discount_factor, globals.nsteps)*next_qvs

        loss = F.mse_loss(believed_qvs, target_v)
        loss2 = F.mse_loss(believed_qvs2, target_v)

        self.sub_agent.opt_step(loss)
        self.sub_agent2.opt_step(loss2)

    def update_target(self):
        self.sub_agent.update_target()
        self.sub_agent2.update_target()
