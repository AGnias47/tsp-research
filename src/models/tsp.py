from typing import Optional
from config import config
import numpy as np

from src.utils.decorators import timing


class TSP:
    """
    Base class for TSP Algorithms. Includes built-in run_tsp method that times the
    algorithm.

    Attributes
    ----------
    algorithm_name: str
        Formal name for repr. Subclass must implement.
    abbreviation: str
        Abbreviated name for calling in main.py. Subclass must implement.
    best_cost: int
        Stores best cost of the algorithm. Initialized to inf.
    best_route: list
        Stores best route found by the algorithm. Initialized to None.
    """

    algorithm_name = NotImplemented
    abbreviation = NotImplemented

    def __init__(self):
        self.best_cost = np.inf
        self.best_route = None
        self.runtime = np.inf

    def __repr__(self):
        return f"{self.algorithm_name} Algorithm"

    @property
    def hyperparameters(self):
        return {}

    @property
    def big_o_runtime(self) -> Optional[None]:
        """
        Estimated runtime units of running an algorithm. Should be implemented as a
        function by the subclass.

        Returns
        -------
        int
            Runtime units. None if not implemented.
        """
        return None

    @timing
    def run_tsp(self):
        """
        Runs the algorithm implemented by the subclass.

        Returns
        -------
        int, list, float
            - Best cost
            - Best route
            - Time to run the algorithm
        """
        self.best_cost, self.best_route = self.algorithm()
        return self.best_cost, self.best_route

    def algorithm(self) -> (int, list[int]):
        """
        Main function for solving TSP. Must be implemented by the subclass.

        Returns
        -------
        int, list
            - Best cost
            - Best route
        """
        raise NotImplementedError
