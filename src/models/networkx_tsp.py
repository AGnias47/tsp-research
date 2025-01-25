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

    def __init__(self, algorithm_name: str, filepath: str):
        """
        Constructor

        Parameters
        ----------
        filepath: str
            Path to file hosting tsplib-formatted problem
        """
        super().__init__(algorithm_name)
        problem = tsplib95.load(filepath)
        self.name = problem.name
        self.G = problem.get_graph()
        self.n = self.G.number_of_nodes()

    def dist(self, i, j):
        return self.G.edges[i, j]["weight"]
