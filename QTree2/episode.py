from dataclasses import dataclass
from typing import *


@dataclass
class Episode:
    states: List[List[float]]
    actions: List[int]
    values: List[List[float]]
    rewards: List[float]
    value_targets: List[float]
    reward_sum: float
