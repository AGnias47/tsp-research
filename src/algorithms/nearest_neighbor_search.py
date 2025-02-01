"""
Nearest neighbor search. Produces a valid TSP path using a greedy algorithm. Not
guaranteed to be optimal.

Used to initialize the pheromone matrix of Ant colony optimization problems.
"""

import networkx.exception
import numpy as np
from networkx import Graph

from src.models.networkx_tsp import NetworkxTSP


class NearestNeighborSearch(NetworkxTSP):
    def __init__(self, filepath):
        super().__init__("Nearest Neighbor Search", filepath)
        self.starting_node = 0

    def algorithm(self):
        S = self.G.copy()
        try:
            S.remove_node(self.starting_node)
        except networkx.exception.NetworkXError:
            self.starting_node = 1
            S.remove_node(1)
        subproblem_cost, subproblem_route = self.subproblem(self.starting_node, S)
        total_cost = subproblem_cost + self.dist(
            subproblem_route[-1], self.starting_node
        )
        final_route = [self.starting_node] + subproblem_route + [self.starting_node]
        return total_cost, np.array(final_route)

    def subproblem(self, source_node, S: Graph):
        if S.number_of_nodes() == 0:
            return 0, []
        else:
            best_cost = np.inf
            best_dest = None
            for dest_node in S.nodes:
                cost = self.dist(source_node, dest_node)
                if cost < best_cost:
                    best_cost = cost
                    best_dest = dest_node
            S.remove_node(best_dest)
            subproblem_cost, subproblem_route = self.subproblem(best_dest, S)
            return best_cost + subproblem_cost, [best_dest] + subproblem_route

    @property
    def big_o_runtime(self):
        return 2 * self.n**2
