"""
Q-Learning Reinforcement Algorithm for solving the Traveling-Salesman problem

References
----------
* https://en.wikipedia.org/wiki/Q-learning - More explicit instructions on Q-table
update
* Watkins. Q-learning. Machine learning, 1992-05, Vol.8 (3-4), p.279-292. 1992,
https://librarysearch.temple.edu/articles/cdi_proquest_miscellaneous_25830152
* https://jamesmccaffrey.wordpress.com/2017/11/30/the-epsilon-greedy-algorithm/ -
e-greedy choice
* https://www.baeldung.com/cs/epsilon-greedy-q-learning - Q-learning description with
examples
"""

from collections import defaultdict

import numpy as np

from config import config
from src.models.networkx_tsp import NetworkxTSP


class QLearning(NetworkxTSP):
    """
    agent - traveler
    environment - cities to visit
    state - cities that have been visited
    action - next city to visit
    reward - distance between cities, reciprocal of cost, negative of cost squared
    Q-table - action values
    action selection - E-greedy strategy
    """

    def __init__(self, filepath):
        super().__init__("Q-Learning", filepath)
        self.alpha = config.q_learning["alpha"]
        self.gamma = config.q_learning["gamma"]
        self.Q = defaultdict(lambda: (0, np.empty(0, dtype=float)))
        self.starting_node = 0 if 0 in self.G else 1
        self.rng = np.random.default_rng(config.random_number_seed)

    @property
    def big_o_runtime(self):
        return None

    @property
    def epsilon(self):
        return 0.999**self.episode  # noqa

    def algorithm(self):
        self.q_learning()
        route = [self.starting_node]
        cost = 0
        while len(route) < self.n:
            action, Q_t = self.next_action(route)
            cost += self.dist(route[-1], action)
            route.append(action)
        cost += self.dist(route[-1], self.starting_node)
        route.append(self.starting_node)
        return cost, route

    def q_learning(self):
        for self.episode in range(config.q_learning["episodes"]):
            S = [self.starting_node]
            while len(S) < self.n:
                action, Q_t = self.next_action(S)
                reward = self.reward(S[-1], action)
                S_t1 = S + [action]
                if len(S_t1) == self.n:
                    a_t1 = self.starting_node
                else:
                    a_t1 = self.next_action(set(self.G.nodes) - set(S_t1))
                Q_t1 = self.Q[(tuple(S_t1), a_t1)]
                temporal_difference_target = reward + self.gamma * Q_t1
                self.Q[(tuple(S), action)] = (
                    1 - self.alpha
                ) * Q_t + self.alpha * temporal_difference_target
                S = S_t1

    def next_action(self, S):
        available_actions = set(self.G.nodes) - set(S)
        if self.rng.random() < self.epsilon:
            return self.explore(S, available_actions)
        else:
            return self.exploit(S, available_actions)

    def explore(self, S, A):
        action = np.random.choice(A)
        return action, self.Q[(tuple(S), action)]

    def exploit(self, S, A):
        max_reward = -np.inf
        a_t1 = None
        for a in A:
            reward = self.Q[(tuple(S), a)]
            if reward > max_reward:
                max_reward = reward
                a_t1 = a
        return a_t1, reward

    def reward(self, i, j):
        return 1 / self.dist(i, j)

    def alt_reward(self, i, j):
        return -(self.dist(i, j) ** 2)
