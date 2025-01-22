from algorithms.np_complete.brute_force import algorithm
from utils.tsplib_parser import tsplib_graph

if __name__ == "__main__":
    G = tsplib_graph("local/data/tsplib/gr17.tsp")
    (best_cost, best_route), total_time = algorithm(G)
    print(f"Best Cost: {best_cost}")
    print(f"Best Route: {best_route}")
    print(f"Total time (s): {total_time}")
