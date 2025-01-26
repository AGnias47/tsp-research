"""
Optimized Brute force solution. Always starts at node 0 and excludes symmetric
routes.

References
----------
* https://stackoverflow.com/a/1985841/8728749 - Method for excluding symmetric permutations
"""

from itertools import permutations

import numpy as np

from src.models.networkx_tsp import NetworkxTSP


class BruteForceOptimized(NetworkxTSP):
    def __init__(self, filepath):
        super().__init__("Brute Force Optimized", filepath)

    def algorithm(self):
        best_cost = np.inf
        best_route = None
        nodes = list(self.G.nodes)[1:]
        for permutation in permutations(nodes):
            if nodes[0] > nodes[-1]:
                continue
            permutation = [0] + list(permutation)
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
