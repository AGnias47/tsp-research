"""
Brute force exact solution to the Traveling Salesman Problem.
"""

from itertools import permutations
from math import factorial

import numpy as np

from src.models.networkx_tsp import NetworkxTSP


class BruteForce(NetworkxTSP):
    algorithm_name = "Brute Force"
    abbreviation = "bf"

    def __init__(self, filepath: str):
        super().__init__(filepath)

    def algorithm(self) -> (int, list[int]):
        """
        Involves iterating through every permutation of routes and calculating the cost.
        The best route is chosen as the one with the lowest cost. Starting and ending
        node is fixed as the first node in the graph.

        Returns
        -------
        int, list
        """
        best_cost = np.inf
        best_route = None
        starting_point = list(self.G.nodes)[0]
        for permutation in permutations(list(self.G.nodes)[1:]):
            permutation = [starting_point] + list(permutation)
            p_cost = float(0)
            p_route = np.empty(self.n + 1, dtype=int)
            p_route[0] = permutation[0]
            for i in range(self.n):
                starting_node = permutation[i]
                ending_node = permutation[(i + 1) % self.n]
                p_cost += self.dist(starting_node, ending_node)
                p_route[i + 1] = ending_node
            p_route[self.n] = p_route[0]
            if p_cost < best_cost:
                best_cost = p_cost
                best_route = p_route
        return best_cost, best_route

    @property
    def big_o_runtime(self) -> int:
        """
        Reduced from O(n!) to O((n-1)!) by fixing the starting point

        Returns
        -------
        int
        """
        return factorial(self.n - 1)
