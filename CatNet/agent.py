import policy
import globals
from typing import *
import torch
import numpy as np
import memory
from transition import *
import torch.nn.functional as F
import math
import random
import utils



class Agent:
    def __init__(self, state_space_size: int, action_space_size: int):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

        self.agents = [policy.Policy(state_space_size, action_space_size).to(globals.device)
                       for _ in range(globals.num_agents)]

    def reset(self, replay_buffer: memory.ReplayMemory):
        self.agents = [policy.Policy(self.state_space_size, self.action_space_size).to(globals.device)
                       for _ in range(globals.num_agents)]

        for i in range(globals.reset_retrain_iters):
            self.improve(replay_buffer.sample(globals.batch_size))

    def get_action(self, state: np.ndarray) -> int:
        agent_to_do_action = random.choice([0, globals.num_agents-1])
        return self.agents[agent_to_do_action].get_action(state)

    def get_expected_values(self, state: np.ndarray) -> torch.tensor:
        agent_to_do_action = random.choice([0, globals.num_agents-1])
        agent = self.agents[agent_to_do_action]
        return agent.get_expected_values(agent.forward_logits(state))

    def improve(self, batch: List[Transition]):
        states = torch.FloatTensor(
            np.array([t.state for t in batch])).to(globals.device)
        next_states = torch.FloatTensor(
            np.array([t.next_state for t in batch])).to(globals.device)
        rewards = torch.FloatTensor(
            [t.reward for t in batch]).to(globals.device)

        # calculate targets
        with torch.no_grad():
            next_all_logits = [agent.forward_logits(next_states)
                              for agent in self.agents]

            next_actions_logits = next_all_logits[0]
            next_actions_values = self.agents[0].get_expected_values(
                next_actions_logits)
            next_actions = next_actions_values.max(1)[1].view(-1, 1)

            next_all_expected_values = [self.agents[i].get_expected_values(
                next_all_logits[i]) for i in range(len(self.agents))]
            heighest_next_qvs = torch.stack(
                [qvs.gather(1, next_actions) for qvs in next_all_expected_values])

            min_next_qvs = heighest_next_qvs.min(0)[0]

            assert (min_next_qvs.size() == heighest_next_qvs[0].size())
            target_v = rewards.view(-1, 1) + \
                math.pow(globals.discount_factor, globals.nsteps)*min_next_qvs
            two_hot_encoded_targets = torch.tensor([utils.get_two_hot(
                v[0], -globals.abs_max_dicrete_value, globals.abs_max_dicrete_value) for v in target_v.tolist()]).to(globals.device)

        for agent in self.agents:
            believed_logits_by_action = agent.forward_logits(states)
            beleived_logits_of_taken_actions = torch.stack([believed_logits_by_action[transition.action][i] for i, transition in enumerate(batch)])
            assert beleived_logits_of_taken_actions.shape == two_hot_encoded_targets.shape, f'{beleived_logits_of_taken_actions.shape} == {two_hot_encoded_targets.shape}'
            loss = F.cross_entropy(beleived_logits_of_taken_actions, two_hot_encoded_targets)
            agent.opt_step(loss)