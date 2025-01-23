"""
Brute force exact solution to the Traveling Salesman Problem
"""

from itertools import permutations
from math import inf

from networkx import Graph

from utils.decorators import timing
from utils.tsplib_parser import tsplib_graph


def algorithm(filepath):
    G = tsplib_graph(filepath)
    return solve(G)


@timing
def solve(G: Graph):
    n = G.number_of_nodes()
    best_cost = inf
    best_route = None
    for permutation in permutations(G.nodes):
        p_cost = 0
        p_route = [permutation[0]]
        for i in range(n):
            starting_node = permutation[i]
            ending_node = permutation[(i + 1) % n]
            p_cost += G.edges[starting_node, ending_node]["weight"]
            p_route.append(ending_node)
        if p_cost < best_cost:
            best_cost = p_cost
            best_route = p_route
    return best_cost, best_route
