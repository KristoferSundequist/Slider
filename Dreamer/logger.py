import matplotlib.pyplot as plt


class Logger:
    def __init__(self):
        self.rewards = []
        self.reconstruction_losses = []
        self.reward_losses = []
        self.transition_losses = []
        self.value_losses = []
        self.policy_losses = []
        self.entropy_losses = []
        self.representation_losses = []

    def add_reward(self, reward: float):
        self.rewards.append(reward)

    def add_reconstuction_loss(self, reconstuction_loss: float):
        self.reconstruction_losses.append(reconstuction_loss)

    def add_reward_loss(self, reward_loss: float):
        self.reward_losses.append(reward_loss)

    def add_transition_loss(self, transition_loss: float):
        self.transition_losses.append(transition_loss)

    def add_value_loss(self, value_loss: float):
        self.value_losses.append(value_loss)

    def add_policy_loss(self, policy_loss: float):
        self.policy_losses.append(policy_loss)

    def add_entropy_loss(self, entropy_loss: float):
        self.entropy_losses.append(entropy_loss)
    
    def add_representation_loss(self, representation_loss: float):
        self.representation_losses.append(representation_loss)

    def get_running_avg_reward(self):
        if len(self.rewards) == 0:
            return 0

        avg_running_reward = self.rewards[0]
        for r in self.rewards[1:]:
            avg_running_reward = avg_running_reward * 0.9 + r * 0.1
        return avg_running_reward

    def plot(self, last_n: int):
        fig, ax = plt.subplots(4, 2)
        ax[0, 0].plot([i for i in range(1, len(self.rewards) + 1)], [number for number in self.rewards])
        ax[0, 0].set(xlabel="#Episode", ylabel="Reward")
        ax[0, 0].grid()

        ax[0, 1].plot(
            [i for i in range(1, len(self.reconstruction_losses) + 1)][-last_n:],
            [number for number in self.reconstruction_losses][-last_n:],
        )
        ax[0, 1].set(xlabel="#Batch", ylabel="Reconstruction loss")
        ax[0, 1].grid()

        ax[1, 0].plot(
            [i for i in range(1, len(self.transition_losses) + 1)][-last_n:],
            [number for number in self.transition_losses][-last_n:],
        )
        ax[1, 0].set(xlabel="#Batch", ylabel="Transition loss")
        ax[1, 0].grid()

        ax[1, 1].plot(
            [i for i in range(1, len(self.reward_losses) + 1)][-last_n:],
            [number for number in self.reward_losses][-last_n:],
        )
        ax[1, 1].set(xlabel="#Batch", ylabel="Reward loss")
        ax[1, 1].grid()

        ax[2, 0].plot(
            [i for i in range(1, len(self.value_losses) + 1)][-last_n:],
            [number for number in self.value_losses][-last_n:],
        )
        ax[2, 0].set(xlabel="#Batch", ylabel="Value loss")
        ax[2, 0].grid()

        ax[2, 1].plot(
            [i for i in range(1, len(self.policy_losses) + 1)][-last_n:],
            [number for number in self.policy_losses][-last_n:],
        )
        ax[2, 1].set(xlabel="#Batch", ylabel="Policy loss")
        ax[2, 1].grid()

        ax[3, 0].plot(
            [i for i in range(1, len(self.entropy_losses) + 1)][-last_n:],
            [number for number in self.entropy_losses][-last_n:],
        )
        ax[3, 0].set(xlabel="#Batch", ylabel="Entropy loss (unscaled)")
        ax[3, 0].grid()

        ax[3, 1].plot(
            [i for i in range(1, len(self.representation_losses) + 1)][-last_n:],
            [number for number in self.representation_losses][-last_n:],
        )
        ax[3, 1].set(xlabel="#Batch", ylabel="Representation loss (unscaled)")
        ax[3, 1].grid()

        plt.show()
