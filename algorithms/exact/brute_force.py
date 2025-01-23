"""
Brute force exact solution to the Traveling Salesman Problem
"""

from itertools import permutations
import numpy as np
from networkx import Graph

from utils.decorators import timing
from utils.tsplib_parser import tsplib_graph


def algorithm(filepath):
    G = tsplib_graph(filepath)
    return solve(G)


@timing
def solve(G: Graph):
    n = G.number_of_nodes()
    best_cost = np.inf
    best_route = None
    for permutation in permutations(G.nodes):
        p_cost = float(0)
        p_route = np.empty(n+1, dtype=int)
        p_route[0] = permutation[0]
        for i in range(n):
            starting_node = permutation[i]
            ending_node = permutation[(i + 1) % n]
            p_cost += G.edges[starting_node, ending_node]["weight"]
            p_route[i] = ending_node
        p_route[n] = p_route[0]
        if p_cost < best_cost:
            best_cost = p_cost
            best_route = p_route
    return best_cost, best_route
