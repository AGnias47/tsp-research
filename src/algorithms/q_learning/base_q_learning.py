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
from types import ModuleType

import numpy as np

from config import config
from src.models.networkx_tsp import NetworkxTSP
from src.utils.figures import plot_costs

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

    def __init__(
        self,
        filepath: str,
        alpha: float,
        gamma: float,
        epsilon_func_key: str,
        reward_func_key: str,
        episodes: int,
        rng: ModuleType = random,
    ):
        """

        Parameters
        ----------
        filepath: str
        alpha: float
            Learning rate
        gamma: float
            Discount factor
        epsilon_func_key: str
            One of e{1,2,3,4}
        reward_func_key: str
            One of r{1,2,3}
        episodes: int
            Number of episodes for Q-Learning
        rng: module
            Random number generator; used in exploratory action choice. Should be able
            to call self.rng.random() and return a random float between 0 and 1
        """
        super().__init__(filepath)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_func_key = epsilon_func_key
        self.reward_func_key = reward_func_key
        self.episodes = episodes
        self.rng = rng
        self.starting_node = 0 if 0 in self.G else 1

    @property
    def big_o_runtime(self) -> int:
        """
        Rough estimate. Similar to dynamic programming methods. More thorough
        algorithmic analysis is needed to definitively give a good estimate of the
        runtime. For example, this is being used for both Q and Double Q-Learning, and
        Double-Q Learning is slower in practice.

        Returns
        -------
        int
            Runtime units
        """
        return self.episodes * self.n**2

    @property
    def epsilon(self) -> int:
        """
        Exploratory factor. When a new action must be decided, a random value between
        0 and 1 is chosen. If it is below epsilon, the next action is randomly chosen,
        else the best next action based on the Q-table is used.

        e1 - decreases linearly
        e2 - concave function
        e3 - convex function
        e4 - step function

        In Wang et al., e1 and e3 generally performed best

        Returns
        -------
        float
        """
        if self.epsilon_func_key == "e1":
            return 1 - self.episode / self.episodes  # noqa
        elif self.epsilon_func_key == "e2":
            return 0.999**self.episode  # noqa
        elif self.epsilon_func_key == "e3":
            return -((self.episode / self.episodes) ** 6) + 1  # noqa
        elif self.epsilon_func_key == "e4":
            return 1 - (0.1 * (self.episode // (self.episodes // 10)))  # noqa
        else:
            raise ValueError(
                "Invalid epsilon function specified in config. Must be e{1,2,3,4}"
            )

    def algorithm(self) -> (int, list[int]):
        costs = self.q_learning()
        if config.debug:
            plot_costs(costs)  # noqa
            self.print_Q_table()
        route = np.array([self.starting_node])
        cost = 0
        state = self.starting_node
        while len(route) < self.n:
            action = self.next_action(
                state, set(self.G.nodes) - set(route), allow_exploration=False
            )
            cost += self.dist(state, action)
            route = np.concatenate((route, np.array([action])))
            state = action
        cost += self.dist(state, self.starting_node)
        route = np.concatenate((route, np.array([self.starting_node])))
        return cost, route

    def q_learning(self) -> list[int]:
        """
        Develops Q Matrix

        Returns
        -------
        list
            Cost for route found in each iteration
        """
        costs = []
        for self.episode in range(self.episodes):
            episode_cost = 0
            episode_starting_node = self.starting_node
            route = np.array([episode_starting_node])
            while len(route) < self.n:
                state = route[-1]
                action = self.next_action(int(state), set(self.G.nodes) - set(route))
                reward = self.reward(int(state), action)
                updated_route = np.concatenate((route, np.array([action])))
                updated_environment = set(self.G.nodes) - set(updated_route)
                if updated_environment:
                    a_t1 = self.next_action(action, updated_environment)
                else:
                    a_t1 = episode_starting_node
                self.update_Q_table(int(state), action, reward, a_t1)
                route = updated_route
                episode_cost += self.dist(route[-2], route[-1])
            state = route[-1]
            action = episode_starting_node
            reward = self.reward(int(state), action)
            a_t1 = self.next_action(
                action, set(self.G.nodes) - set(np.array([episode_starting_node]))
            )
            self.update_Q_table(int(state), action, reward, a_t1)
            episode_cost += self.dist(int(route[-1]), episode_starting_node)
            costs.append(episode_cost)
        return costs

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

    def exploit(self, s: int, environment: set[int]) -> int:
        """
        Next action to choose based on values provided in Q-table.

        Parameters
        ----------
        s: int
            Node value representing the current state
        environment: set
            Available nodes to travel to

        Raises
        ------
        NotImplementedError
            If the function is not implemented in the inheriting class

        Returns
        -------
        int
            Ideal node to travel to based on value in Q-table
        """
        raise NotImplementedError

    def next_action(
        self, state: int, environment: set[int], allow_exploration: bool = True
    ) -> int:
        """
        Decides the next action for the agent to take. Utilizes a random float and
        epsilon to determine if a random destination should be chosen (explore) or if
        the knowledge from the Q-table should be used (exploit).

        Parameters
        ----------
        state: int
            Starting node
        environment: set
            Available nodes to travel to
        allow_exploration: bool (defaults to True)
            If exploration should be allowed. Should be set to True during Q-Learning
            and False when generating the optimal route.

        Returns
        -------
        int
            Destination node
        """
        if self.rng.random() < self.epsilon and allow_exploration:
            return self.explore(environment)
        else:
            return self.exploit(state, environment)

    @staticmethod
    def explore(environment: set[int]) -> int:
        """
        Chooses a random node from the environment

        Parameters
        ----------
        environment: list
            Available nodes to travel to

        Returns
        -------
        int
        """
        return random.choice(list(environment))

    def reward(self, i: int, j: int) -> float:
        """
        Reward function, either penalizing for long routes or rewarding for short
        routes. Functions taken from Wang et al.

        r1 - reciprocal of distance, shorter distances will be larger
        r2 - negative of distance
        r3 - negative of distance squared

        Generally r1 was found to have the best average performance in Wang et al.

        Parameters
        ----------
        i: start
        j: end

        Raises
        ------
        ValueError
            If an invalid reward function is specified in the config

        Returns
        -------
        float
        """
        if self.reward_func_key == "r1":
            try:
                return 1 / self.dist(i, j)
            except ZeroDivisionError as e:
                # If these are actually 2 different cities, return best possible reward
                if abs(i - j) > 0:
                    return 10
                # Else this is a bug, raise error
                print(i, j)
                raise e
        if self.reward_func_key == "r2":
            return -self.dist(i, j)
        if self.reward_func_key == "r3":
            return -(self.dist(i, j) ** 2)
        raise ValueError(
            "Invalid reward function specified in config. Must be r{1,2,3}"
        )

    def print_Q_table(self):
        """
        Prints the contents of the Q table(s)

        Returns
        -------
        None
        """
        if hasattr(self, "Q"):
            for k, v in self.Q.items():
                print(f"{k[0]} | {k[1]} | {v}")
        if hasattr(self, "Q_a") and hasattr(self, "Q_b"):
            print("------------------")
            print("Q_a")
            print("------------------")
            for k, v in self.Q_a.items():
                print(f"{k[0]} | {k[1]} | {v}")
            print("------------------")
            print("Q_b")
            print("------------------")
            for k, v in self.Q_b.items():
                print(f"{k[0]} | {k[1]} | {v}")
