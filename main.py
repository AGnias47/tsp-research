"""
Main runscript

References
----------
* https://stackoverflow.com/a/10742904/8728749 - Formatting large numbers
"""

import argparse

import tsplib95

from src.algorithms.ant_system import AntSystem
from src.algorithms.brute_force import BruteForce
from src.algorithms.concorde import Concorde
from src.algorithms.held_karp import HeldKarp
from src.algorithms.nearest_neighbor_search import NearestNeighborSearch
from src.utils.arg_parsing import get_filepath_for_problem

ALGORITHMS = {
    "concorde": Concorde,
    "nns": NearestNeighborSearch,
    "hk": HeldKarp,
    "as": AntSystem,
    "bf": BruteForce,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--problem",
        required=True,
        help="Problem to run, either by index, problem name, or path",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        required=False,
        help="Algorithm to use. If not specified, problem is run on all available algorithms",
    )
    args = parser.parse_args()
    filepath = get_filepath_for_problem(args.problem)
    if args.algorithm:
        algorithm_list = []
        for algo in args.algorithm.split(","):
            try:
                algorithm_list.append(ALGORITHMS[algo])
            except KeyError:
                parser.error(f"Invalid algorithm specified: {algo}")
    else:
        algorithm_list = ALGORITHMS.values()
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
    for algorithm in algorithm_list:
        solver = algorithm(filepath)
        print(f"Results of the {solver}")
        if solver.big_o_runtime:
            print(f"Runtime units: {solver.big_o_runtime:_}")
        (best_cost, best_route), total_time = solver.run_tsp()
        print(f"Best Cost: {best_cost}")
        print(f"Best Route: {best_route}")
        print(f"Time to Solve: {total_time}")
        print("-----------------------")
