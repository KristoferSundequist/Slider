import torch

discount_factor = 0.99
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nsteps = 3
updates_per_step = 1
batch_size = 32
replay_memory_size = 500000
reset_freq = 80
learning_rate = 1e-4
epsilon = 0.1
reset_retrain_iters = 50000
target_update_freq = 3000
num_agents = 2

width = 700
height = 700