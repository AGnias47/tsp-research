"""
Main runscript

References
----------
* https://stackoverflow.com/a/10742904/8728749 - Formatting large numbers
"""

import argparse

import tsplib95

from src.algorithms.brute_force import BruteForce
from src.algorithms.concorde import Concorde
from src.algorithms.held_karp import HeldKarp

PROBLEMS = [
    "custom/p5.tsp",  # 0
    "papers/barachet10.tsp",  # 1
    "custom/p11.tsp",  # 2
    "custom/p12.tsp",  # 3
    "tsplib/ulysses16.tsp",  # 4
    "tsplib/gr17.tsp",  # 5
    "tsplib/fri26.tsp",  # 6
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--problem",
        required=True,
        help="Problem to run by index in the PROBLEMS list",
        type=int,
    )
    args = parser.parse_args()
    filepath = f"local/data/{PROBLEMS[args.problem]}"
    name = tsplib95.load(filepath).name
    print(f"Solutions for the {name} problem")
    print("-----------------------")
    for algorithm in [Concorde, HeldKarp, BruteForce]:
        solver = algorithm(filepath)
        print(f"Results of the {solver}")
        if solver.big_o_runtime:
            print(f"Runtime units: {solver.big_o_runtime:_}")
        (best_cost, best_route), total_time = solver.run_tsp()
        print(f"Best Cost: {best_cost}")
        print(f"Best Route: {best_route}")
        print(f"Time to Solve: {total_time}")
        print("-----------------------")
