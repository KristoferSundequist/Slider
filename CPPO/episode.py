from dataclasses import dataclass
from typing import *
from TwoHotEncodingDistribution import TwoHotEncodingDistribution
import torch


class Episode:
    def __init__(self):
        self.states: List[List[float]] = []
        self.actions: List[List[float]] = []
        self.value_logits: List[List[float]] = []
        self.rewards: List[float] = []
        self.action_means: List[List[float]] = []
        self.action_stds: List[List[float]] = []

    def get_reward_sum(self) -> float:
        return sum(self.rewards)

    def get_value_targets(self, discount_factor: float, lambd: float) -> List[float]:
        targets = [0.0 for _ in range(len(self.rewards))]
        values = TwoHotEncodingDistribution(torch.tensor(self.value_logits)).mean.squeeze().tolist()
        targets[-1] = values[-1]

        for i in reversed(range(len(values) - 1)):
            bootstrap = (1.0 - lambd) * values[i + 1] + lambd * targets[i + 1]
            targets[i] = self.rewards[i] + discount_factor * bootstrap
        return targets

    def add_transition(
        self,
        state: List[float],
        action: List[float],
        value_logits: List[float],
        reward: float,
        action_means: List[float],
        action_stds: List[float],
    ):
        self.states.append(state)
        self.actions.append(action)
        self.value_logits.append(value_logits)
        self.rewards.append(reward)
        self.action_means.append(action_means)
        self.action_stds.append(action_stds)
