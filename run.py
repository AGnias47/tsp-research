import tsplib95

from src.algorithms.concorde.concorde_wrapper import Concorde
from src.algorithms.exact.brute_force import BruteForce
from src.algorithms.exact.held_karp import HeldKarp

PROBLEMS = [
    "custom/p5.tsp",  # 0
    "papers/barachet10.tsp",  # 1
    "custom/p11.tsp",  # 2
    "custom/p12.tsp",  # 3
    "tsplib/ulysses16.tsp",  # 4
    "tsplib/gr17.tsp",  # 5
]

if __name__ == "__main__":
    problem = PROBLEMS[1]
    filepath = f"local/data/{problem}"
    name = tsplib95.load(filepath).name
    print(f"Solutions for the {name} problem")
    print("-----------------------")
    for algorithm in [Concorde, HeldKarp, BruteForce]:
        solver = algorithm(filepath)
        (best_cost, best_route), total_time = solver.run_tsp()
        print(f"Results of the {solver}")
        print(f"Best Cost: {best_cost}")
        print(f"Best Route: {best_route}")
        print(f"Time to Solve: {total_time}")
        print("-----------------------")
