"""
Brute force NP-complete solution to the Traveling Salesman Problem
"""

from networkx import Graph
from math import inf

def algorithm(G: Graph):
    salesman_home = G[0]
    min_cost = inf
    route = None
    for X in G.nodes[1:]:
        pass
        # https://en.wikipedia.org/wiki/Heap%27s_algorithm
