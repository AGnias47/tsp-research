from algorithms.tsp import TSP
from utils.tsplib_parser import tsplib_graph


class NetworkxTSP(TSP):
    def __init__(self, filepath):
        self.filepath = filepath
        self.G = tsplib_graph(filepath)
        self.n = self.G.number_of_nodes()

    def dist(self, i, j):
        return self.G.edges[i, j]["weight"]
