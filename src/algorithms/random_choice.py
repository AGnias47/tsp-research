"""
Randomly chooses a route.

Generates 1000 random routes and chooses the best one.
"""

import random

import numpy as np
import tsplib95

from src.models.networkx_tsp import NetworkxTSP


class RandomChoice(NetworkxTSP):
    algorithm_name = "Random Choice"
    abbreviation = "r"

    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.name = tsplib95.load(filepath).name

    def algorithm(self) -> (int, list[int]):
        best_cost = np.inf
        best_route = None
        starting_point = list(self.G.nodes)[0]
        nodes = list(self.G.nodes)[1:]
        for _ in range(10_000):
            random.shuffle(nodes)
            p_cost, p_route = self.permutation_cost([starting_point] + nodes)
            if p_cost < best_cost:
                best_cost = p_cost
                best_route = p_route
        return best_cost, best_route
