"""
Dynamic Programming exact solution to the Traveling Salesman Problem. Implemented from Held-Karp's "A Dynamic
Programming Approach to Sequencing Problems".

Uses a memoization matrix, D. Doesn't seem to have a noticeable impact on runtime, but good to have if distance
function ever gets more complex.

Resources
---------
* https://stackoverflow.com/a/46151546/8728749 - initializing an infinity matrix
"""

import numpy as np
from networkx import Graph

from src.models.networkx_tsp import NetworkxTSP


class DP(NetworkxTSP):
    def __init__(self, filepath):
        super().__init__("Held-Karp", filepath)
        self.starting_node = 0
        self.D = np.matrix(np.ones((self.n, self.n)) * np.inf)

    def algorithm(self):
        S = self.G.copy()
        S.remove_node(self.starting_node)
        best_cost = np.inf
        best_route = None
        for l in S.nodes:
            S_cost, S_route = self.dp_subproblem(S, l)
            if self.D[l, 0] == np.inf:
                self.D[l, 0] = self.dist(l, 0)
            cost = S_cost + self.D[l, 0]
            route = np.concatenate((S_route, np.array([l])))
            if cost < best_cost:
                best_cost = cost
                best_route = np.concatenate((np.array([0]), route, np.array([0])))
        return best_cost, best_route

    def dp_subproblem(self, S: Graph, l: int):
        if S.number_of_nodes() == 1:
            if self.D[self.starting_node, l] == np.inf:
                self.D[self.starting_node, l] = self.dist(self.starting_node, l)
            return self.D[self.starting_node, l], np.empty(0, dtype=int)
        else:
            S_min_l = S.copy()
            S_min_l.remove_node(l)
            best_cost = np.inf
            best_route = None
            for m in S_min_l.nodes():
                if self.D[m, l] == np.inf:
                    self.D[m, l] = self.dist(m, l)
                S_cost, S_route = self.dp_subproblem(S_min_l, m)
                cost = S_cost + self.D[m, l]
                if cost < best_cost:
                    best_cost = cost
                    best_route = np.concatenate((S_route, np.array([m])))
            return best_cost, best_route
