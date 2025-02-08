import numpy as np

from src.algorithms.brute_force import BruteForce
from src.algorithms.concorde import Concorde


def test_brute_force():
    filepath = "data/custom/p5.tsp"
    (concorde_best_cost, concorde_best_route), concorde_runtime = Concorde(
        filepath,
        suppress_output=False,  # Context manager gives issues in pytest, probably not ideal, use with caution
    ).run_tsp()
    (brute_force_best_cost, brute_force_best_route), brute_force_runtime = BruteForce(
        filepath
    ).run_tsp()
    assert concorde_best_cost == brute_force_best_cost
    assert type(concorde_best_route) is type(brute_force_best_route) is np.ndarray
    assert type(concorde_runtime) is type(brute_force_runtime) is float
