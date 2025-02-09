"""
Main runscript

References
----------
* https://stackoverflow.com/a/10742904/8728749 - Formatting large numbers
"""

import argparse

import tsplib95

from config import config
from src.algorithms.aco.ant_system import AntSystem
from src.algorithms.aco.max_min_ant_system import MaxMinAntSystem
from src.algorithms.brute_force import BruteForce
from src.algorithms.concorde import Concorde
from src.algorithms.held_karp import HeldKarp
from src.algorithms.nearest_neighbor_search import NearestNeighborSearch
from src.algorithms.q_learning.double_q_learning import DoubleQLearning
from src.algorithms.q_learning.q_learning import QLearning
from src.utils.arg_parsing import get_filepath_for_problem

ALGORITHMS = {
    "concorde": Concorde,
    "nns": NearestNeighborSearch,
    "hk": HeldKarp,
    "as": AntSystem,
    "mmas": MaxMinAntSystem,
    "bf": BruteForce,
    "q": QLearning,
    "dq": DoubleQLearning,
    "ants": [AntSystem, MaxMinAntSystem],
    "proj": [Concorde, MaxMinAntSystem, DoubleQLearning],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--problem",
        required=True,
        help="Problem to run, either by problem name, index, or absolute path. "
        f"Valid names include: {list(config.problems.keys())}. "
        f"See config.yaml::problems for path information and to add new problem names.",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        required=False,
        help="Algorithm to use. If not specified, problem is run on all available "
        f"algorithms. Valid algorithm names include: {list(ALGORITHMS.keys())}. "
        f"See main.py::ALGORITHMS for references to each algorithm name.",
    )
    args = parser.parse_args()
    problems = []
    for problem in args.problem.split(","):
        filepath = get_filepath_for_problem(problem)
        try:
            name = tsplib95.load(filepath).name
        except FileNotFoundError:
            parser.error(
                f"{filepath} is not a valid path. "
                "Specify a problem by its index or problem name in config.yaml::problems, "
                "or manually specify the path to the problem."
            )
        problems.append((filepath, name))
    if args.algorithm:
        algorithm_list = []
        for algo in args.algorithm.split(","):
            try:
                algorithm_choice = ALGORITHMS[algo]
                if isinstance(algorithm_choice, list):
                    algorithm_list.extend(algorithm_choice)
                else:
                    algorithm_list.append(algorithm_choice)
            except KeyError:
                parser.error(
                    f"Invalid algorithm specified: {algo}. "
                    f"Must be one of, or a comma-separated list including only: "
                    f"{list(ALGORITHMS.keys())}"
                )
    else:
        algorithm_list = ALGORITHMS.values()
    for filepath, name in problems:
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
