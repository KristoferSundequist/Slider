import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
from torch import autograd, nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

torch.manual_seed(1)

lstm = nn.GRU(3, 3)  # Input dim is 3, output dim is 3
inputs = [Variable(torch.randn(2, 3)) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = Variable(torch.randn(1, 2, 3), volatile=True)

for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    print(hidden)
    out, hidden = lstm(i.view(1, -1, 3), hidden)
