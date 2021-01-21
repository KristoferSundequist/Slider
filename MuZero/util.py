import torch
import numpy as np

'''

    ONE HOT

'''

def onehot(size: int, i: int) -> torch.Tensor:
    return torch.eye(size)[i]

def test_onehot_equal_simple():
    x = onehot(4, 2)
    assert torch.equal(x, torch.Tensor([0,0,1,0]))

def test_onehot_not_equal():
    x = onehot(4, 2)
    assert not torch.equal(x, torch.Tensor([1,0,0,0]))

'''

    normalize minmax

'''

def normalize_minmax_tensor(v):
    return (v-v.min()).true_divide((v.max() - v.min()))

def test_normalize_tensor():
    t = torch.tensor([[-20,0, 3, 5]])
    nt = normalize_minmax_tensor(t)
    assert nt.shape == t.shape
    assert nt.min().item() == 0
    assert nt.max().item() == 1

'''

    Discretize

'''


def discretize(v: float, n_buckets: int = 41, abs_max_v: float = 2) -> np.ndarray:
    clipped_v = np.clip(v, -abs_max_v, abs_max_v)
    discrete_value = int(clipped_v*10 + 20)
    return onehot(n_buckets, discrete_value)

def test_discretize_zero():
    v = discretize(0, 41, 2)
    assert v.sum() == 1
    assert v[20] == 1

def test_discretize_maxneg():
    v = discretize(-2, 41, 2)
    assert v.sum() == 1
    assert v[0] == 1

def test_discretize_maxpos():
    v = discretize(2, 41, 2)
    assert v.sum() == 1
    assert v[40] == 1

def test_discretize_little_pos():
    v = discretize(0.3, 41, 2)
    assert v.sum() == 1
    assert v[23] == 1

def test_discretize_outofbounds():
    v = discretize(-3, 41, 2)
    assert v.sum() == 1
    assert v[0] == 1

def test_discretize_outofbounds():
    v = discretize(5, 41, 2)
    assert v.sum() == 1
    assert v[40] == 1

'''

    expected_value

'''

def expected_value(probs, abs_max_v: float = 2):
    values = torch.linspace(-abs_max_v, abs_max_v, probs.shape[1])
    return torch.einsum("i, bi -> b", values, probs)

def test_expected_value():
    probs = torch.tensor([[0.2,0.3,0.1,0.3,0.1], [0.1,0.1,0.3,0.4,0.1]])
    v = expected_value(probs, 1)
    assert round(v[0].item(),2) == round(0.2*(-1) + 0.3*(-0.5) + 0.1*0 + 0.3*0.5 + 0.1*1, 2)

    assert round(v[1].item(),2) == round(0.1*(-1) + 0.1*(-0.5) + 0.3*0 + 0.4*0.5 + 0.1*1, 2)


'''

    Normalizer

'''

class Normalizer:
    def __init__(self, known_min: float, known_max: float):
        self.minV = known_min
        self.maxV = known_max
    
    def update(self, value: float):
        self.minV = min(self.minV, value)
        self.maxV = max(self.maxV, value)
    
    def get_normalized(self, value: float):
        return (value - self.minV)/(self.maxV - self.minV)


def test_normalizer():
    norm = Normalizer(-2, 2)

    assert norm.get_normalized(2) == 1
    assert norm.get_normalized(1) == 0.75
    assert norm.get_normalized(-2) == 0
    assert norm.get_normalized(0) == 0.5

    norm.update(3)

    assert norm.get_normalized(3) == 1
    assert norm.get_normalized(2) == 0.8
    assert norm.get_normalized(-2) == 0

    norm.update(-4)

    assert norm.get_normalized(-4) == 0
    assert norm.get_normalized(-0.5) == 0.5

'''

    Categotical cross entropy

'''

def categorical_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()

def test_categorical_cross_entropy():
    a = torch.tensor([[0.2,0.1,0.5,0.2]])
    b = torch.tensor([[0.7,0.1,0.1,0.1]])
    c = torch.tensor([[0.2,0.2,0.4,0.2]])
    loss_ab = categorical_cross_entropy(a,b)
    loss_ac = categorical_cross_entropy(a,c)
    assert loss_ab > loss_ac

def test_categorical_cross_entropy():
    a = torch.tensor([[0.2,0.1,0.5,0.2]])
    b = torch.tensor([[0.7,0.1,0.1,0.1]])

    c = torch.tensor([[0.2,0.2,0.4,0.2]])
    d = torch.tensor([[0.1,0.1,0.1,0.7]])
    loss_ac = categorical_cross_entropy(a,c)
    loss_bd = categorical_cross_entropy(b,d)
    
    ab = torch.cat([a,b])
    cd = torch.cat([c,d])
    loss_ac_bd = categorical_cross_entropy(ab,cd)

    assert (loss_ac + loss_bd)/2 == loss_ac_bd