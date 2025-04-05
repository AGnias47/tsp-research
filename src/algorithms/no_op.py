"""
Provides the known optimum cost from the config
"""

import tsplib95

from config import config
from src.models.tsp import TSP


class NoOp(TSP):
    algorithm_name = "No-Op"
    abbreviation = "noop"

    def __init__(self, filepath: str):
        super().__init__()
        self.name = tsplib95.load(filepath).name

    def algorithm(self) -> (int, list[int]):
        if optimum_cost := config.optimum_costs.get(self.name):
            return optimum_cost, []
        raise RuntimeError(f"No optimum cost found for {self.name}")
