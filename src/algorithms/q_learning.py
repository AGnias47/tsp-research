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
import random
import numpy as np
import matplotlib.pyplot as plt
from config import config
from src.models.networkx_tsp import NetworkxTSP


random.seed(config.random_number_seed)


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
        self.Q = defaultdict(lambda: 0)
        self.starting_node = 0 if 0 in self.G else 1
        self.rng = random

    @property
    def big_o_runtime(self):
        return None

    @property
    def epsilon(self):
        if config.q_learning["episodes"] < 8_000:
            return 1 - self.episode / config.q_learning["episodes"]  # noqa
        else:
            return 0.999**self.episode  # noqa

    def algorithm(self):
        best_cost = np.inf
        best_route = None
        costs = []
        for self.episode in range(config.q_learning["episodes"]):
            episode_cost = 0
            S = [self.starting_node]
            while len(S) < self.n:
                action, Q_t = self.next_action(S)
                reward = self.reward(S[-1], action)
                S_t1 = S + [action]
                if len(S_t1) == self.n:
                    a_t1 = self.starting_node
                    Q_t1 = self.Q[(action, a_t1)]
                else:
                    a_t1, Q_t1 = self.next_action(S_t1)
                temporal_difference_target = reward + self.gamma * Q_t1 - Q_t
                self.Q[S[-1], action] = (
                    Q_t + self.alpha * temporal_difference_target
                )
                S = S_t1
                episode_cost += self.dist(S[-2], S[-1])
            episode_cost += self.dist(S[-1], self.starting_node)
            if episode_cost < best_cost:
                best_cost = episode_cost
                best_route = S + [self.starting_node]
            costs.append(episode_cost)
        self.plot_costs(costs)
        self.print_Q_table()
        return best_cost, best_route

    def next_action(self, S, allow_exploration=True):
        available_actions = set(self.G.nodes) - set(S)
        if self.rng.random() < self.epsilon and allow_exploration:
            return self.explore(S, available_actions)
        else:
            return self.exploit(S, available_actions)

    def explore(self, S, A):
        action = random.choice(list(A))
        return action, self.Q[S[-1], action]

    def exploit(self, S, A):
        max_reward = -np.inf
        a_t1 = None
        for a in A:
            reward = self.Q[S[-1], a]
            if reward > max_reward:
                max_reward = reward
                a_t1 = a
        return a_t1, reward  # noqa

    def reward(self, i, j):
        if self.n < 1:
            return -(self.dist(i, j) ** 2)
        else:
            return 1 / self.dist(i, j)

    def use_q_table(self):
        costs = self.algorithm()
        self.plot_costs(costs)
        route = [self.starting_node]
        cost = 0
        while len(route) < self.n:
            action, Q_t = self.next_action(route, allow_exploration=False)
            cost += self.dist(route[-1], action)
            route.append(action)
        cost += self.dist(route[-1], self.starting_node)
        route.append(self.starting_node)
        return cost, route

    def print_Q_table(self):
        for k, v in self.Q.items():
            print(f"{k[0]} | {k[1]} | {v}")

    def plot_costs(self, costs):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        ax.set_title("Cost over each RL episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cost")
        ax.plot(costs)
        plt.show()
