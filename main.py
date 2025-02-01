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
from src.utils.arg_parsing import get_filepath_for_problem

PROBLEMS = {
    "p5": "custom/p5.tsp",  # 0
    "barachet10": "papers/barachet10.tsp",  # 1
    "p11": "custom/p11.tsp",  # 2
    "p12": "custom/p12.tsp",  # 3
    "ulysses16": "tsplib/ulysses16.tsp",  # 4
    "gr17": "tsplib/gr17.tsp",  # 5
    "fri26": "tsplib/fri26.tsp",  # 6
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--problem",
        required=True,
        help="Problem to run, either by index, problem name, or path",
    )
    args = parser.parse_args()
    filepath = get_filepath_for_problem(PROBLEMS, args.problem)
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
