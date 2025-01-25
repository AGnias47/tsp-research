import numpy as np

from src.algorithms.concorde import concorde_wrapper
from src.algorithms import brute_force


def test_brute_force():
    filepath = "local/data/custom/p5.tsp"
    (concorde_best_cost, concorde_best_route), concorde_runtime = (
        concorde_wrapper.algorithm(filepath)
    )
    (brute_force_best_cost, brute_force_best_route), brute_force_runtime = (
        brute_force.algorithm(filepath)
    )
    assert concorde_best_cost == brute_force_best_cost
    assert type(concorde_best_cost) == type(brute_force_best_cost) == float
    assert type(concorde_best_route) == type(brute_force_best_route) == np.ndarray
    assert type(concorde_runtime) == type(brute_force_runtime) == float
