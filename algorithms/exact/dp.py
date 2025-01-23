"""
Dynamic Programming exact solution to the Traveling Salesman Problem
"""

from math import inf

from networkx import Graph

from utils.decorators import timing


@timing
def algorithm(G: Graph):
    n = G.number_of_nodes()
