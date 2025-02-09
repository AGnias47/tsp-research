import random
from collections import defaultdict

import numpy as np

from config import config
from src.algorithms.q_learning.base_q_learning import BaseQLearning

random.seed(config.random_number_seed)


class DoubleQLearning(BaseQLearning):
    def __init__(self, filepath):
        super().__init__(filepath, "Double Q-Learning")
        self.Q_a = defaultdict(lambda: 0)
        self.Q_b = defaultdict(lambda: 0)

    def update_Q_table(self, state, action, reward, a_t1):
        Q_a_t1 = self.Q_a[action, a_t1]
        Q_b_t1 = self.Q_b[action, a_t1]
        tdt_a = reward + self.gamma * Q_b_t1 - self.Q_a[state, action]
        tdt_b = reward + self.gamma * Q_a_t1 - self.Q_b[state, action]
        self.Q_a[state, action] = self.Q_a[state, action] + self.alpha * tdt_a
        self.Q_b[state, action] = self.Q_b[state, action] + self.alpha * tdt_b

    def exploit(self, s, E):
        max_reward = -np.inf
        a_t1 = None
        for a in E:
            reward = np.average([self.Q_a[s, a], self.Q_b[s, a]])
            if reward > max_reward:
                max_reward = reward
                a_t1 = a
        return a_t1
