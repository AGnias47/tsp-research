"""
Solves TSP via Concorde. Used as a source of truth for finding the Optimal Tour Value.

References
----------
* https://www.math.uwaterloo.ca/tsp/concorde.html - Concorde homepage
"""

import numpy as np
import tsplib95
from concorde.tsp import TSPSolver  # noqa

from src.models.tsp import TSP
from src.utils.context_managers import redirect_output_to_null


class Concorde(TSP):
    algorithm_name = "Concorde"
    abbreviation = "concorde"

    def __init__(self, filepath: str, suppress_output: bool = True):
        """
        Constructor.

        Parameters
        ----------
        filepath: str
            Full path to problem
        suppress_output: bool (defaults to True)
            If True, sends output from Concorde executable to /dev/null. Else, prints
            to STD{OUT,ERR}.
        """
        super().__init__()
        self.name = tsplib95.load(filepath).name
        self.suppress_output = suppress_output
        if self.suppress_output:
            with redirect_output_to_null():
                self.concorde_solver = TSPSolver.from_tspfile(filepath)
        else:
            self.concorde_solver = TSPSolver.from_tspfile(filepath)

    def algorithm(self) -> (int, list[int]):
        """
        Returns the result of the Concorde sovler.

        Returns
        -------

        """
        if self.suppress_output:
            with redirect_output_to_null():
                solution = self.concorde_solver.solve()
        else:
            solution = self.concorde_solver.solve()
        return solution.optimal_value, np.concatenate(
            (solution.tour, np.array([solution.tour[0]]))
        )
