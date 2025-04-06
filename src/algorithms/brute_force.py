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
            p_cost, p_route = self.permutation_cost(
                [starting_point] + list(permutation)
            )
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
