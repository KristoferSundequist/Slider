import torch

discount_factor = 0.99
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
replay_buffer_size = 500000

world_model_learning_rate = 6e-4
value_model_learning_rate = 8e-5
action_model_learning_rate = 8e-5

hidden_vector_size = 256

sequence_length = 50
batch_size = 51
imagination_horizon=15
entropy_coeff = 0.005

update_frequency = 20

width = 700
height = 700