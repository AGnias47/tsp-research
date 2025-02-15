from collections import defaultdict

import numpy as np

from src.algorithms.q_learning.base_q_learning import BaseQLearning


class QLearning(BaseQLearning):
    algorithm_name = "Q-Learning"
    abbreviation = "q"

    def __init__(self, filepath):
        super().__init__(filepath)
        self.Q = defaultdict(lambda: 0)

    def update_Q_table(self, state, action, reward, a_t1):
        Q_t = self.Q[state, action]
        Q_t1 = self.Q[action, a_t1]
        temporal_difference_target = reward + self.gamma * Q_t1 - Q_t
        self.Q[state, action] = Q_t + self.alpha * temporal_difference_target

    def exploit(self, s, environment):
        max_reward = -np.inf
        a_t1 = None
        for a in environment:
            reward = self.Q[s, a]
            if reward > max_reward:
                max_reward = reward
                a_t1 = a
        return a_t1
