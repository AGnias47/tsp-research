"""
Main runscript

References
----------
* https://stackoverflow.com/a/10742904/8728749 - Formatting large numbers
"""

import argparse
import sys

import mlflow

from config import config
from src.algorithms.aco.ant_system import AntSystem
from src.algorithms.aco.max_min_ant_system import MaxMinAntSystem
from src.algorithms.brute_force import BruteForce
from src.algorithms.concorde import Concorde
from src.algorithms.held_karp import HeldKarp
from src.algorithms.nearest_neighbor_search import NearestNeighborSearch
from src.algorithms.q_learning.double_q_learning import DoubleQLearning
from src.algorithms.q_learning.q_learning import QLearning
from src.utils.arg_parsing import get_available_problems, get_filepath_for_problem
from src.utils.mlflow_client import log_results

ALGORITHMS = [
    Concorde,
    NearestNeighborSearch,
    BruteForce,
    HeldKarp,
    AntSystem,
    MaxMinAntSystem,
    QLearning,
    DoubleQLearning,
]
ALGORITHM_DICT = {a.abbreviation: a for a in ALGORITHMS} | {
    "aco": [AntSystem, MaxMinAntSystem],
    "rl": [QLearning, DoubleQLearning],
    "proj": [Concorde, MaxMinAntSystem, DoubleQLearning],
    "all": ALGORITHMS,
}

sys.setrecursionlimit(100_000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--problem",
        required=True,
        help="Problem to run, either by problem name, index, or absolute path. "
        f"Valid names include: {get_available_problems()}. "
        f"See config.yaml::problems for path information and to add new problem names.",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        required=False,
        help="Algorithm to use. If not specified, problem is run on all available "
        f"algorithms. Valid algorithm names include: {list(ALGORITHM_DICT.keys())}. "
        f"See main.py::ALGORITHM_DICT for mapping to each algorithm name.",
    )
    parser.add_argument(
        "-l", "--log-results", action="store_true", help="Logs results to MLflow"
    )
    parser.add_argument(
        "-m",
        "--mlflow-project",
        required=False,
        help="MLflow Project name. "
        "log-results must be active for this to have any effect.",
    )
    args = parser.parse_args()
    problems = []
    for problem in args.problem.split(","):
        filepath = get_filepath_for_problem(problem)
        if not filepath:
            parser.error(
                f"Could not find {problem}.tsp within the "
                f"{config.problems_parent_path} directory."
            )
        problems.append((filepath, problem))
    if args.algorithm:
        algorithm_list = []
        for algo in args.algorithm.split(","):
            try:
                algorithm_choice = ALGORITHM_DICT[algo]
                if isinstance(algorithm_choice, list):
                    algorithm_list.extend(algorithm_choice)
                else:
                    algorithm_list.append(algorithm_choice)
            except KeyError:
                parser.error(
                    f"Invalid algorithm specified: {algo}. "
                    f"Must be one of, or a comma-separated list including only: "
                    f"{list(ALGORITHM_DICT.keys())}"
                )
    else:
        algorithm_list = algorithm_choice = ALGORITHM_DICT["proj"]
    for filepath, name in problems:
        if args.log_results:
            if args.mlflow_project:
                mlflow.set_experiment(args.mlflow_project)
            else:
                mlflow.set_experiment("TSP Project")
        print(f"Solutions for the {name} problem")
        print("-----------------------")
        for algorithm in algorithm_list:
            solver = algorithm(filepath)
            print(f"Results of the {solver}")
            if solver.big_o_runtime:
                print(f"Runtime units: {solver.big_o_runtime:_}")
            (cost, route), total_time = solver.run_tsp()
            print(f"Cost: {cost}")
            print(f"Route: {route}")
            print(f"Time to Solve: {total_time}")
            print("-----------------------")
            if args.log_results:
                solver.best_cost = cost
                solver.best_route = route
                solver.runtime = total_time
                log_results(name, solver)
