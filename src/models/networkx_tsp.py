import numpy as np
import tsplib95

from src.models.tsp import TSP


class NetworkxTSP(TSP):
    """
    Base class for TSP problems using a Networkx Graph as the data structure storing Nodes and Edges

    Attributes
    ----------
    name: str
        Name of the problem being solved
    G: networkx.Graph
        Representation of the problem as a graph where cities are nodes and connections are weighted edges
    n: int
        Number of nodes in the graph
    """

    def __init__(self, filepath: str):
        """
        Constructor

        Parameters
        ----------
        filepath: str
            Path to file hosting tsplib-formatted problem
        """
        super().__init__()
        problem = tsplib95.load(filepath)
        self.name = problem.name
        self.G = problem.get_graph()
        self.n = self.G.number_of_nodes()

    def dist(self, i: int, j: int) -> int:
        """
        Cost of traveling from node i to node j

        Parameters
        ----------
        i: int
            Index of starting node
        j: int
            Index of ending node

        Returns
        -------
        int
            Cost from traveling from i to j
        """
        return self.G.edges[i, j]["weight"]

    def permutation_cost(self, permutation):
        p_cost = float(0)
        p_route = np.empty(self.n + 1, dtype=int)
        p_route[0] = permutation[0]
        for i in range(self.n):
            starting_node = permutation[i]
            ending_node = permutation[(i + 1) % self.n]
            p_cost += self.dist(starting_node, ending_node)
            p_route[i + 1] = ending_node
        p_route[self.n] = p_route[0]
        return p_cost, p_route
