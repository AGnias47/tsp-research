from utils.decorators import timing
from utils.tsplib_parser import tsplib_graph


class TSP:
    def __init__(self, filepath):
        self.filepath = filepath
        self.G = tsplib_graph(filepath)
        self.n = self.G.number_of_nodes()

    @timing
    def run_tsp(self):
        return self.algorithm()

    def algorithm(self):
        raise NotImplementedError

    def dist(self, i, j):
        return self.G.edges[i, j]["weight"]
