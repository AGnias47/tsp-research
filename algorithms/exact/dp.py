"""
Dynamic Programming exact solution to the Traveling Salesman Problem

Resources
---------
* https://stackoverflow.com/a/46151546/8728749 - initializing an infinity matrix
"""

from networkx import Graph
import numpy as np
from utils.decorators import timing
from utils.tsplib_parser import tsplib_graph


starting_node = 0


def solve(S: Graph, l: int, D: np.matrix[int]):
    n = S.number_of_nodes()
    if n == 1:
        if D[starting_node, l] == np.inf:
            D[starting_node, l] = S.edges[starting_node, l]["weight"]
        return D[starting_node, l], np.array([l]), D
    else:
        S.remove_node(l)
        best_cost = np.inf
        best_route = None
        S.remove_node(l)
        for m in S.nodes():
            if D[m, l] == np.inf:
                D[m, l] = S.edges[m, l]["weight"]
            S_cost, S_route, D = solve(S, m, D)
            cost = S_cost + D[m, l]
            if cost < best_cost:
                best_cost = cost
                best_route = np.concatenate(np.array([m]), S_route)
        return best_cost, best_route, D


@timing
def algorithm(filepath):
    G = tsplib_graph(filepath)
    n = G.number_of_nodes()
    D = np.matrix(np.ones((n, n)) * np.inf)
    S = G.copy()
    S.remove_node(starting_node)
    best_cost = np.inf
    best_route = np.empty(n+1, dtype=int)
    for l in G.nodes:
        S_cost, S_route, D = solve(S, l, D)
        if D[l, 0] == np.inf:
            D[l, 0] = G.edges[l, 0]["weight"]
        cost = S_cost + D[l, 0]
        route = np.concatenate(S_route, np.array([0]))
        if cost < best_cost:
            best_cost = cost
            best_route = route
    return best_cost, best_route
