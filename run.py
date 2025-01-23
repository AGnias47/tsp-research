from algorithms.concorde import concorde_wrapper
from algorithms.exact import brute_force

if __name__ == "__main__":
    filepath = "local/data/custom/p5.tsp"
    for library in [concorde_wrapper, brute_force]:
        (best_cost, best_route), total_time = library.algorithm(filepath)
        print("-----------------------")
        print(f"Best Cost: {best_cost}")
        print(f"Best Route: {best_route}")
        print(f"Total time (s): {total_time}")
