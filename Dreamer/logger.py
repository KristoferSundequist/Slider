import matplotlib.pyplot as plt
import math


class Logger:
    def __init__(self):
        self.stuff = {}

    def add(self, key: str, number: float):
        if self.stuff.get(key) is None:
            self.stuff[key] = []
        self.stuff[key].append(number)

    def get_avg_of_window(self, key: str, size: int):
        if self.stuff.get(key) is None:
            return None

        return sum(self.stuff[key][-size:]) / size

    def get_running_avg(self, key: str):
        if self.stuff.get(key) is None:
            return None

        things = self.stuff[key]
        avg_running = things[0]

        for t in things[1:]:
            avg_running = avg_running * 0.9 + t * 0.1
        return avg_running

    def plot(self, last_n: int):
        n_plots = len(self.stuff.keys())
        n_columns = 4
        fig, ax = plt.subplots(math.ceil(n_plots / n_columns), n_columns)

        for i, key in enumerate(self.stuff.keys()):
            row = math.floor(i / n_columns)
            column = i % n_columns

            ax[row, column].plot(
                [i for i in range(1, len(self.stuff[key]) + 1)][-last_n:],
                [number for number in self.stuff[key]][-last_n:],
            )
            ax[row, column].set(xlabel="Tick", ylabel=key)
            ax[row, column].grid()

        plt.show()
