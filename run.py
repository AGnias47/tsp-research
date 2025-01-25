from src.algorithms.concorde.concorde_wrapper import Concorde
from src.algorithms.exact.brute_force import BruteForce
from src.algorithms.exact.dp import DP

if __name__ == "__main__":
    filepath = "local/data/papers/barachet10.tsp"
    for algorithm in [Concorde, DP, BruteForce]:
        solver = algorithm(filepath)
        (best_cost, best_route), total_time = solver.run_tsp()
        print("-----------------------")
        print(f"Best Cost: {best_cost}")
        print(f"Best Route: {best_route}")
        print(f"Time to Solve: {total_time}")
