"""
Brute force exact solution to the Traveling Salesman Problem
"""

from itertools import permutations
from math import factorial

import numpy as np

from src.models.networkx_tsp import NetworkxTSP


class BruteForce(NetworkxTSP):
    def __init__(self, filepath):
        super().__init__("Brute Force", filepath)

    def algorithm(self):
        best_cost = np.inf
        best_route = None
        for permutation in permutations(self.G.nodes):
            p_cost = 0
            p_route = np.empty(self.n + 1, dtype=int)
            p_route[0] = permutation[0]
            for i in range(self.n):
                starting_node = permutation[i]
                ending_node = permutation[(i + 1) % self.n]
                p_cost += self.dist(starting_node, ending_node)
                p_route[i] = ending_node
            p_route[self.n] = p_route[0]
            if p_cost < best_cost:
                best_cost = p_cost
                best_route = p_route
        return best_cost, best_route

    @property
    def big_o_runtime(self):
        return factorial(self.n)
