"""
References
----------
* https://stackoverflow.com/a/569063/8728749 - zero matrix
* https://stackoverflow.com/a/55507797/8728749 - efficient initialization of a
matrix to a single value in each element
"""

from numpy import np

from src.models.networkx_tsp import NetworkxTSP
from src.algorithms.nearest_neighbor_search import NearestNeighborSearch


class ACO(NetworkxTSP):
    """
    Ant colony optimization algorithm for the traveling-salesman problem. Ants act as
    agents that construct tours. Tours are guided by pheromone trails.

    Initially, m ants are placed on random cities. At each city, a state transition rule
    is applied. Trails that are close and with high pheromone strength are
    probabilistically preferred. Each ant has a tabu list which stores the partial tour.

    Once all ants have completed a tour, the pheromones are updated. This is done by
    lowering the pheromone trail strengths by a constant and then allowing the ants to
    deposit pheromone on the arcs they have visited. Arcs contained in shorter tours and
    that have been visited by many ants receive a higher pheromone update.
    """

    def __init__(self, filepath, apply_local_search=False):
        super().__init__("Ant Colony Optimization", filepath)
        self.apply_local_search = apply_local_search
        # Number of ants
        self.m = 1
        # Tracks pheromone strength. Initialize using the algorithm m/C^nn, where m is
        #   number of ants and C^nn is the length of a tour generated via nearest
        #   neighbor search. Initializing too low causes bias in early tours, and too
        #   high increases time to convergence.
        nn_cost, _ = NearestNeighborSearch(filepath)
        self.tau = np.ndarray(shape=(self.n, self.n))
        self.tau[:] = self.m / nn_cost
        # Cache heuristic information
        self.eta = np.zeros(shape=(self.n, self.n))
        # Hyperparameters
        self.alpha = 1
        self.beta = 1

    def algorithm(self):
        for _ in range(1000):
            self.construct_solutions()
            if self.apply_local_search:
                self.local_search()
            self.update_trails()

    def probabilistic_action_choice(self, ant, i, j):
        tau_ij = self.tau[i, j]
        eta_ij = self.heuristic(i, j)
        numerator = tau_ij**self.alpha * eta_ij**self.beta
        denominator = 0
        for l in self.G.nodes - ant.tabu_list:
            tau_il = self.tau[i, l]
            eta_il = self.heuristic(i, l)
            denominator += tau_il**self.alpha * eta_il**self.beta
        return numerator / denominator

    def heuristic(self, i, j):
        """
        Heuristic desirability of going from city i to j

        Parameters
        ----------
        i: int
            Starting node
        j: int
            Ending node

        Returns
        -------
        np.float
        """
        if self.eta[i, j] > 0:
            return self.eta[i, j]
        try:
            self.eta[i, j] = 1 / self.dist(i, j)
        except ZeroDivisionError:
            self.eta[i, j] = 1e-9
        return self.eta[i, j]

    def construct_solutions(self):
        pass

    def update_trails(self):
        pass

    def local_search(self):
        pass
