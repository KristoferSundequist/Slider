import torch

discount_factor = 0.99
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
replay_buffer_size = 500000

world_model_learning_rate = 1e-4
world_model_adam_eps = 1e-8
actor_critic_learning_rate = 3e-5
actor_critic_adam_eps = 1e-5

mlp_size = 256
stoch_vector_size = 255
recurrent_vector_size = 256

sequence_length = 64
batch_size = 16
imagination_horizon=15
entropy_coeff = 0.005

update_frequency = 20

width = 700
height = 700