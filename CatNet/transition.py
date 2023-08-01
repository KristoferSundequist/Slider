from dataclasses import dataclass
import numpy as np

@dataclass
class Transition:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
