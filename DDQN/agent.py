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

        self.agents = [Sub_Agent(state_space_size, action_space_size)
                       for _ in range(globals.num_agents)]

    def reset(self, replay_buffer: memory.ReplayMemory):
        for agent in self.agents:
            agent.reset()

        for i in range(globals.reset_retrain_iters):
            self.improve(replay_buffer.sample(globals.batch_size))
            if i % 2000 == 0:
                for agent in self.agents:
                    agent.update_target()

    def get_action(self, state: np.ndarray) -> int:
        agent_to_do_action = random.choice([0, globals.num_agents-1])
        return self.agents[agent_to_do_action].get_action(state)

    def get_values(self, state: np.ndarray) -> torch.tensor:
        agent_to_do_action = random.choice([0, globals.num_agents-1])
        return self.agents[agent_to_do_action].get_values(state)

    def improve(self, batch: List[Transition]):
        states = torch.FloatTensor(
            np.array([t.state for t in batch])).to(globals.device)
        actions = torch.LongTensor(
            [t.action for t in batch]).to(globals.device)
        next_states = torch.FloatTensor(
            np.array([t.next_state for t in batch])).to(globals.device)
        rewards = torch.FloatTensor(
            [t.reward for t in batch]).to(globals.device)

        believed_qvs = [agent.forward(states).gather(
            1, actions.view(-1, 1)) for agent in self.agents]

        # double dqn
        with torch.no_grad():
            next_all_qvs = [agent.forward(next_states)
                            for agent in self.agents]

            next_actions = next_all_qvs[0].max(1)[1].view(-1, 1)

            heighest_qvs = torch.stack(
                [qvs.gather(1, next_actions) for qvs in next_all_qvs])

            min_next_qvs = heighest_qvs.min(0)[0]

            assert (min_next_qvs.size() == heighest_qvs[0].size())
            target_v = rewards.view(-1, 1) + \
                math.pow(globals.discount_factor, globals.nsteps)*min_next_qvs

        for i in range(0, globals.num_agents):
            loss = F.mse_loss(believed_qvs[i], target_v)
            self.agents[i].opt_step(loss)

    def update_target(self):
        for agent in self.agents:
            agent.update_target()
