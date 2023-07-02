class Counter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count = self.count + 1

batch_counter = Counter()
main_counter = Counter()

class Logger:
    def __init__(self):
        self.head_losses: [(float, float, float, float, float)] = []
        self.rewards: [float] = []
    
    def get_mean_rewards_of_last_n(self, n: int):
        last_n_rewards = self.rewards[-n:]
        sum_rewards = sum(last_n_rewards)
        return sum_rewards/len(last_n_rewards)

    def add_head_losses(self, total_loss, policy_loss, value_loss, reward_loss, sim_loss):
        self.head_losses.append((total_loss, policy_loss, value_loss, reward_loss, sim_loss))