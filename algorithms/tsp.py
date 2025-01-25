import numpy as np

from utils.decorators import timing


class TSP:
    def __init__(self):
        self.best_cost = np.inf
        self.best_route = None

    @timing
    def run_tsp(self):
        self.best_cost, self.best_route = self.algorithm()
        return self.best_cost, self.best_route

    def algorithm(self):
        raise NotImplementedError
