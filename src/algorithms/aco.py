"""
References
----------
* https://stackoverflow.com/a/569063/8728749 - zero matrix
"""

from src.models.networkx_tsp import NetworkxTSP
from numpy import np
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
        # Tracks pheromone strength
        self.tau = np.zeros(shape=(self.n, self.n))
        # Number of ants
        self.m = 1
        self.alpha = 1
        self.beta = 1

    def algorithm(self):
        for _ in range(1000):
            self.construct_solutions()
            if self.apply_local_search:
                self.local_search()
            self.update_trails()

    def probabilistic_action_choice(self, ant, start, end):
        numerator = self.tau[start,end]**self.alpha * (1/self.dist(start, end))**self.beta
        denominator = 0
        for l in (self.G.nodes - ant.tabu_list):
            denominator += self.tau[start,l]**self.alpha * (1/self.dist(start, l))**self.beta
        return numerator / denominator

    def construct_solutions(self):
        pass

    def update_trails(self):
        pass

    def local_search(self):
        pass