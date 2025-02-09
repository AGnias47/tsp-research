"""
Q-Learning Reinforcement Learning Algorithm for solving the Traveling-Salesman Problem.
Adapted from:

Wang, J., Xiao, C., Wang, S.,
Ruan, Y.: Reinforcement learning for the traveling
salesman problem: Performance comparison of three
algorithms. J. Eng. 2023, e12303 (2023).
https://doi.org/10.1049/tje2.12303

References
----------
* https://en.wikipedia.org/wiki/Q-learning - More explicit instructions on Q-table
update
* https://jamesmccaffrey.wordpress.com/2017/11/30/the-epsilon-greedy-algorithm/ -
e-greedy choice
* https://www.baeldung.com/cs/epsilon-greedy-q-learning - Q-learning description with
examples
* https://github.com/mehdibnc/TSP-Q-Learning-/blob/master/src/utils.py - Implementation
reference. Inspired plotting and using state as current city in Q-table
"""

import random

import matplotlib.pyplot as plt
import numpy as np

from config import config
from src.models.networkx_tsp import NetworkxTSP

random.seed(config.random_number_seed)


class BaseQLearning(NetworkxTSP):
    """
    Base Library for solving the Traveling-Salesman Problem using Q-Learning.

    Agent - Salesman
    Environment - Cities left to visit
    State - Current City of the Agent
    Action - Next City to visit
    Reward - Value representative of distance between two Cities, where closer Cities
    give a higher Reward

    Requires
    --------
    update_Q_table - Function that updates the Q-table
    exploit - Function for deciding the next action
    """

    def __init__(self, filepath, name="Q-Learning"):
        super().__init__(name, filepath)
        # learning rate
        self.alpha = config.q_learning["alpha"]
        # discount factor
        self.gamma = config.q_learning["gamma"]
        self.starting_node = 0 if 0 in self.G else 1
        # random number generator; used in exploratory action choice. Should be able to
        #   call self.rng.random() and return a random float between 0 and 1
        self.rng = random

    @property
    def big_o_runtime(self):
        """
        Rough estimate. Similar to dynamic programming methods. More thorough
        algorithmic analysis is needed to definitively give a good estimate of the
        runtime.

        Returns
        -------
        int
        """
        return config.q_learning["episodes"] * self.n**2

    @property
    def epsilon(self) -> int:
        """
        Exploratory factor. When a new action must be decided, a random value between
        0 and 1 is chosen. If it is below epsilon, the next action is randomly chosen,
        else the best next action based on the Q-table is used.

        For problems where less than 8,000 episodes are used, an epsilon starting at 1
        with linear decay is used. Else, an exponential function is used that was
        discovered to be effective in Wang et al. Doesn't start to truly utilize the
        Q-table until episode > 7,500.

        Returns
        -------
        float
        """
        if config.q_learning["episodes"] < 8_000:
            return 1 - self.episode / config.q_learning["episodes"]  # noqa
        else:
            return 0.999**self.episode  # noqa

    def update_Q_table(self, state: int, action: int, reward: float, a_t1: int) -> None:
        """
        Once an action and next best action are chosen, update the Q-table for the
        current state and action.

        Parameters
        ----------
        state: int
            Current location of the Agent
        action: int
            Where the Agent will travel to next
        reward: float
            Reward for traveling from state to action
        a_t1: int
            Best destination from action, i.e. ideal action in the next state

        Raises
        ------
        NotImplementedError
            If the function is not implemented in the inheriting class

        Returns
        -------
        None
        """
        raise NotImplementedError

    def exploit(self, s, environment):
        """
        Next action to choose based on values provided in Q-table.

        Parameters
        ----------
        s: int
            Node value representing the current state
        environment: list
            Available nodes to travel to

        Returns
        -------
        int
            Ideal node to travel to based on value in Q-table
        """
        raise NotImplementedError

    def algorithm(self):
        costs = self.q_learning()
        if config.debug:
            self.plot_costs(costs)
            self.print_Q_table()
        route = [self.starting_node]
        cost = 0
        state = self.starting_node
        while len(route) < self.n:
            action = self.next_action(state, set(self.G.nodes) - set(route), allow_exploration=False)
            cost += self.dist(state, action)
            route.append(action)
            state = action
        cost += self.dist(state, self.starting_node)
        route.append(self.starting_node)
        return cost, route

    def q_learning(self):
        costs = []
        for self.episode in range(config.q_learning["episodes"]):
            episode_cost = 0
            route = [self.starting_node]
            while len(route) < self.n:
                state = route[-1]
                action = self.next_action(state, set(self.G.nodes) - set(route))
                reward = self.reward(state, action)
                updated_route = route + [action]
                updated_environment = set(self.G.nodes) - set(updated_route)
                if updated_environment:
                    a_t1 = self.next_action(action, updated_environment)
                else:
                    a_t1 = self.starting_node
                self.update_Q_table(state, action, reward, a_t1)
                route = updated_route
                episode_cost += self.dist(route[-2], route[-1])
            episode_cost += self.dist(route[-1], self.starting_node)
            costs.append(episode_cost)
        return costs

    def next_action(self, state, environment, allow_exploration=True):
        if self.rng.random() < self.epsilon and allow_exploration:
            return self.explore(environment)
        else:
            return self.exploit(state, environment)

    @staticmethod
    def explore(environment):
        return random.choice(list(environment))

    def reward(self, i, j):
        if self.n < 1:
            return -(self.dist(i, j) ** 2)
        else:
            return 1 / self.dist(i, j)

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
