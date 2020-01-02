import torch

def onehot(size: int, i: int) -> torch.Tensor:
    return torch.eye(size)[i]

def test_onehot_equal_simple():
    x = onehot(4, 2)
    assert torch.equal(x, torch.Tensor([0,0,1,0]))

def test_onehot_not_equal():
    x = onehot(4, 2)
    assert not torch.equal(x, torch.Tensor([1,0,0,0]))
