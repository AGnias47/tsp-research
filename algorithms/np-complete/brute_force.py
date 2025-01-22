"""
Brute force NP-complete solution to the Traveling Salesman Problem
"""

from itertools import permutations
from math import inf

from networkx import Graph

from utils.decorators import timing
from utils.tsplib_parser import tsplib_graph


@timing
def algorithm(G: Graph):
    best_cost = inf
    best_route = None
    node_count = G.number_of_nodes()
    for permutation in permutations(G.nodes):
        p_cost = 0
        p_route = [permutation[0]]
        for i in range(node_count):
            starting_node = permutation[i]
            ending_node = permutation[(i + 1) % node_count]
            p_cost += G.edges[starting_node, ending_node]["weight"]
            p_route.append(ending_node)
        if p_cost < best_cost:
            print("New best solution found")
            print(f"Best Cost: {best_cost}")
            print(f"Best Route: {best_route}")
            best_cost = p_cost
            best_route = p_route
    return best_cost, best_route


if __name__ == "__main__":
    G = tsplib_graph("local/data/tsplib/gr17.tsp")
    (best_cost, best_route), total_time = algorithm(G)
    print(f"Best Cost: {best_cost}")
    print(f"Best Route: {best_route}")
    print(f"Total time (s): {total_time}")
