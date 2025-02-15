"""
Dynamic Programming exact solution to the Traveling Salesman Problem. Implemented from
Held-Karp's "A Dynamic Programming Approach to Sequencing Problems".

Uses a memoization matrix, D. Doesn't seem to have a noticeable impact on runtime,
but good to have if distance function ever gets more complex.

Resources
---------
* https://stackoverflow.com/a/46151546/8728749 - initializing an infinity matrix
* https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm - reference for pseudocode
* https://github.com/CarlEkerot/held-karp/blob/master/held-karp.py - Inspiration to use
  cost instead of distance memoization
* https://stackoverflow.com/a/51660293/8728749 - Using tuples as dict keys
* https://stackoverflow.com/a/8483900/8728749 - Initializing a defaultdict with tuples
"""

from collections import defaultdict

import networkx.exception
import numpy as np
from networkx import Graph

from src.models.networkx_tsp import NetworkxTSP


class HeldKarp(NetworkxTSP):
    algorithm_name = "Held-Karp"
    abbreviation = "hk"

    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.starting_node = 0
        self.D = defaultdict(lambda: (-1, np.empty(0, dtype=int)))

    def algorithm(self) -> (int, list[int]):
        S = self.G.copy()
        try:
            S.remove_node(self.starting_node)
        except networkx.exception.NetworkXError:
            self.starting_node = 1
            S.remove_node(1)
        best_cost = np.inf
        best_route = None
        for l in list(S.nodes)[:-1]:
            cost, route = self.D[(tuple(S.nodes), l)]
            if cost < 0:
                cost, route = self.dp_subproblem(S, l)
                cost += self.dist(l, self.starting_node)
                route = np.concatenate((route, np.array([l])))
            if cost < best_cost:
                best_cost = cost
                best_route = np.concatenate(
                    (
                        np.array([self.starting_node]),
                        route,
                        np.array([self.starting_node]),
                    )
                )
        return best_cost, best_route

    def dp_subproblem(self, S: Graph, l: int) -> (int, list[int]):
        cost, route = self.D[(tuple(S.nodes), l)]
        if route.size > 0:
            return cost, route
        if S.number_of_nodes() == 1:
            self.D[(tuple(S.nodes), l)] = self.dist(self.starting_node, l), np.empty(
                0, dtype=int
            )
            return self.D[(tuple(S.nodes), l)]
        else:
            S_min_l = S.copy()
            S_min_l.remove_node(l)
            best_cost = np.inf
            best_route = None
            for m in S_min_l.nodes():
                cost, route = self.dp_subproblem(S_min_l, m)
                cost += self.dist(m, l)
                if cost < best_cost:
                    best_cost = cost
                    best_route = np.concatenate((route, np.array([m])))
            self.D[(tuple(S.nodes), l)] = best_cost, best_route  # noqa
            return best_cost, best_route

    @property
    def big_o_runtime(self) -> int:
        return self.n**2 * 2**self.n
