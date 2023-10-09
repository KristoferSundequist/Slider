from dataclasses import dataclass
from typing import *

@dataclass
class Episode:
    states: List[List[float]]
    actions: List[List[float]]
    values: List[float]
    rewards: List[float]
    action_means: List[List[float]]
    action_stds: List[List[float]]
    value_targets: List[float]
    reward_sum: float