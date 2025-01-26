import tsplib95

from src.algorithms.concorde.concorde_wrapper import Concorde
from src.algorithms.exact.brute_force import BruteForce
from src.algorithms.exact.brute_force_optimized import BruteForceOptimized
from src.algorithms.exact.dp import DP

if __name__ == "__main__":
    problems = ["custom/p5.tsp", "papers/barachet10.tsp"]
    for problem in problems:
        filepath = f"local/data/{problem}"
        name = tsplib95.load(filepath).name
        print(f"Solutions for the {name} problem")
        print("-----------------------")
        for algorithm in [Concorde, DP, BruteForce, BruteForceOptimized]:
            solver = algorithm(filepath)
            (best_cost, best_route), total_time = solver.run_tsp()
            print(f"Results of the {solver}")
            print(f"Best Cost: {best_cost}")
            print(f"Best Route: {best_route}")
            print(f"Time to Solve: {total_time}")
            print("-----------------------")
