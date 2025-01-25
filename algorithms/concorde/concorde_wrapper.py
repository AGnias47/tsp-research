"""
Solves TSP via Concorde. Used as a source of truth for finding the Optimal Tour Value
for custom-generated tours.
"""

from concorde.tsp import TSPSolver  # noqa

from algorithms.tsp import TSP


class Concorde(TSP):
    def __init__(self, filepath):
        self.concorde_solver = TSPSolver.from_tspfile(filepath)

    def algorithm(self):
        solution = self.concorde_solver.solve()
        return solution.optimal_value, solution.tour
