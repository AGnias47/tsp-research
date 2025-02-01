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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--problem",
        required=True,
        help="Problem to run, either by index, problem name, or path",
    )
    args = parser.parse_args()
    filepath = get_filepath_for_problem(args.problem)
    try:
        name = tsplib95.load(filepath).name
    except FileNotFoundError:
        parser.error(
            f"{filepath} is not a valid path. "
            "Specify a problem by its index or problem name in config.yaml::problems, "
            "or manually specify the path to the problem."
        )
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
