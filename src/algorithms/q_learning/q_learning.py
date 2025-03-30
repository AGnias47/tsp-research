import numpy as np

from config import config
from src.algorithms.q_learning.base_q_learning import BaseQLearning


class QLearning(BaseQLearning):
    algorithm_name = "Q-Learning"
    abbreviation = "q"

    def __init__(
        self,
        filepath: str,
        alpha: float = config.q_learning["alpha"],
        gamma: float = config.q_learning["gamma"],
        epsilon_func_key: str = config.q_learning["epsilon"],
        reward_func_key: str = config.q_learning["reward"],
        episodes: int = config.q_learning["episodes"],
    ):
        super().__init__(
            filepath=filepath,
            alpha=alpha,
            gamma=gamma,
            epsilon_func_key=epsilon_func_key,
            reward_func_key=reward_func_key,
            episodes=episodes,
        )
        self.Q = np.zeros(shape=(self.n + 1, self.n + 1))

    @property
    def hyperparameters(self):
        return config.q_learning

    def update_Q_table(self, state: int, action: int, reward: float, a_t1: int) -> None:
        """
        References
        ----------
        * https://huggingface.co/learn/deep-rl-course/en/unit3/deep-q-algorithm
        """
        Q_t = self.Q[state, action]
        Q_t1 = self.Q[action, a_t1]
        temporal_difference_target = reward + self.gamma * Q_t1
        temporal_difference_error = temporal_difference_target - Q_t
        self.Q[state, action] = Q_t + self.alpha * temporal_difference_error

    def exploit(self, s: int, environment: set[int]) -> int:
        max_reward = -np.inf
        a_t1 = None
        for a in environment:
            reward = self.Q[s, a]
            if reward > max_reward:
                max_reward = reward
                a_t1 = a
        return a_t1
