"""
Solves TSP via Concorde. Used as a source of truth for finding the Optimal Tour Value
for custom-generated tours.
"""

import numpy as np
import tsplib95
from concorde.tsp import TSPSolver  # noqa

from src.models.tsp import TSP
from src.utils.context_managers import redirect_output_to_null


class Concorde(TSP):
    def __init__(self, filepath, suppress_output=True):
        super().__init__("Concorde")
        self.name = tsplib95.load(filepath).name
        self.suppress_output = suppress_output
        if self.suppress_output:
            with redirect_output_to_null():
                self.concorde_solver = TSPSolver.from_tspfile(filepath)
        else:
            self.concorde_solver = TSPSolver.from_tspfile(filepath)

    def algorithm(self):
        if self.suppress_output:
            with redirect_output_to_null():
                solution = self.concorde_solver.solve()
        else:
            solution = self.concorde_solver.solve()
        return solution.optimal_value, np.concatenate(
            (solution.tour, np.array([solution.tour[0]]))
        )
