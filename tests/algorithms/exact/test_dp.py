"""
Pytest for Held-Karp algorithm

Resources
---------
* https://stackoverflow.com/a/46914500/8728749 - Testing equality of two lists
"""

from src.algorithms.exact.dp import DP
import tsplib95
import pytest

problem_file = "local/data/tsplib/ulysses16.tsp"
tour_file = "local/data/tsplib/ulysses16.opt.tour"


@pytest.fixture
def expected_results():
    problem = tsplib95.load(problem_file)
    tour = tsplib95.load(tour_file)
    expected_cost = problem.trace_tours(tour.tours)[0]
    expected_route = tour.tours[0]
    return {"cost": expected_cost, "route": expected_route}


def test_dp(expected_results):
    (cost, route), runtime = DP(problem_file).run_tsp()
    assert cost == expected_results["cost"]
    assert all([a == b for a, b in zip(route[:-1], expected_results["route"])])
    assert type(runtime) == float
