"""
Solves TSP via Concorde. Used as a source of truth for finding the Optimal Tour Value
for custom-generated tours.
"""

from concorde.tsp import TSPSolver  # noqa

from utils.decorators import timing


def algorithm(filepath: str):
    solver = TSPSolver.from_tspfile(filepath)
    return solve(solver)


@timing
def solve(solver):
    solution = solver.solve()
    return solution.optimal_value, solution.tour
