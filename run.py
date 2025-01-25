from algorithms.concorde.concorde_wrapper import Concorde
from algorithms.exact.brute_force import BruteForce

if __name__ == "__main__":
    filepath = "local/data/papers/barachet10.tsp"
    for algorithm in [Concorde, BruteForce]:
        solver = algorithm(filepath)
        (best_cost, best_route), total_time = solver.run_tsp()
        print("-----------------------")
        print(f"Best Cost: {best_cost}")
        print(f"Best Route: {best_route}")
