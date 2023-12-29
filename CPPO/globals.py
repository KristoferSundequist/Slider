import torch

discount_factor = 0.99
lambd_factor = 0.95
pvo_epochs = 10
max_grad_norm = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1024
update_clamp_threshold = 0.2
entropy_coef = 0.02
two_hot_bins = 22