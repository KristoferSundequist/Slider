import torch

discount_factor = 0.99
lambd_factor = 0.95
pvo_epochs = 3
max_grad_norm = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
update_clamp_threshold = 0.2