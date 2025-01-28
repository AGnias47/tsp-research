import numpy as np

from src.utils.decorators import timing


class TSP:
    """
    Base class for TSP Algorithms. Includes built-in run_tsp method that times the algorithm.
    """

    def __init__(self, algorithm_name):
        self.algorithm_name = algorithm_name
        self.best_cost = np.inf
        self.best_route = None

    def __repr__(self):
        return f"{self.algorithm_name} Algorithm"

    @timing
    def run_tsp(self):
        self.best_cost, self.best_route = self.algorithm()
        return self.best_cost, self.best_route

    def algorithm(self):
        raise NotImplementedError

    @property
    def big_o_runtime(self):
        return None
