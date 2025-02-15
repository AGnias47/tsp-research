import random

import numpy as np

from config import config
from src.algorithms.q_learning.base_q_learning import BaseQLearning

random.seed(config.random_number_seed)


class DoubleQLearning(BaseQLearning):
    algorithm_name = "Double Q-Learning"
    abbreviation = "dq"

    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.Q_a = np.zeros(shape=(self.n + 1, self.n + 1))
        self.Q_b = np.zeros(shape=(self.n + 1, self.n + 1))

    def update_Q_table(self, state: int, action: int, reward: float, a_t1: int) -> None:
        Q_a_t1 = self.Q_a[action, a_t1]
        Q_b_t1 = self.Q_b[action, a_t1]
        tdt_a = reward + self.gamma * Q_b_t1 - self.Q_a[state, action]
        tdt_b = reward + self.gamma * Q_a_t1 - self.Q_b[state, action]
        self.Q_a[state, action] = self.Q_a[state, action] + self.alpha * tdt_a
        self.Q_b[state, action] = self.Q_b[state, action] + self.alpha * tdt_b

    def exploit(self, s: int, environment: set[int]) -> int:
        max_reward = -np.inf
        a_t1 = None
        for a in environment:
            reward = np.average([self.Q_a[s, a], self.Q_b[s, a]])
            if reward > max_reward:
                max_reward = reward
                a_t1 = a
        return a_t1
