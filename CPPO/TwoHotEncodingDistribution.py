import torch
from torch.distributions import Distribution
from torch import Tensor
import torch.nn.functional as F

class TwoHotEncodingDistribution(Distribution):
    def __init__(self, logits: Tensor, low: int = -10, high: int = 10):
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1)
        self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)

    @property
    def mean(self) -> Tensor:
        return (self.probs * self.bins).sum(dim=1, keepdim=True)

    @property
    def mode(self) -> Tensor:
        return (self.probs * self.bins).sum(dim=1, keepdim=True)

    def log_prob(self, x: Tensor) -> Tensor:
        below = (self.bins <= x).type(torch.int32).sum(dim=-1, keepdim=True) - 1
        above = len(self.bins) - (self.bins > x).type(torch.int32).sum(dim=-1, keepdim=True)
        below = torch.clip(below, 0, len(self.bins) - 1)
        above = torch.clip(above, 0, len(self.bins) - 1)
        equal = below == above
        dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, len(self.bins)) * weight_below[..., None]
            + F.one_hot(above, len(self.bins)) * weight_above[..., None]
        ).squeeze(-2)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        return (target * log_pred).sum(dim=1)