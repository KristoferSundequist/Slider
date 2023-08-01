import matplotlib.pyplot as plt

class Logger:
    def __init__(self):
        self.rewards = []
    
    def add_reward(self, reward: int):
        self.rewards.append(reward)

    def get_running_avg_reward(self):
        if len(self.rewards) == 0:
            return 0
        
        avg_running_reward = self.rewards[0]
        for r in self.rewards[1:]:
            avg_running_reward = avg_running_reward * 0.9 + r * 0.1
        return avg_running_reward
    
    def plot(self):
        fig, ax = plt.subplots()
        ax.plot([i for i in range(1, len(self.rewards)+1)], [r for r in self.rewards])
        ax.set(xlabel='episode', ylabel='reward')
        ax.grid()
        plt.show()
