import torch

def hook(grad):
    new_grad = grad/2
    print(new_grad)
    return new_grad

def testfn():
    x = torch.tensor([3.0], requires_grad=True)
    x.register_hook(hook)
    y = x * 2
    x2 = x + 10
    x2.register_hook(hook)
    y += x2 * 2

    
    y.backward()
    print(x.grad)