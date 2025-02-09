"""
Generic algorithm for Ant Colony Optimization Algorithms, such as Ant System or 
Max-Min Ant System. Adapted from Dorigo and Stützle - Ant Colony Optimization Chapter 3. 
Function descriptions are summarizations of that text.

By default, implements Ant System, as most ACO algorithms are either this or an
expansion upon Ant System. Updates can be made by changing arg inputs and overriding
the default methods.

References
----------
* https://web2.qatar.cmu.edu/~gdicaro/15382/additional/aco-book.pdf - text of Ant Colony
Optimization textbook
* https://stackoverflow.com/a/569063/8728749 - zero matrix
* https://stackoverflow.com/a/55507797/8728749 - efficient initialization of a
matrix to a single value in each element
* https://stackoverflow.com/a/21088294/8728749 - np list to array
"""

import numpy as np

from config import config
from src.algorithms.nearest_neighbor_search import NearestNeighborSearch
from src.models.ant import Ant
from src.models.networkx_tsp import NetworkxTSP
from src.utils.figures import plot_costs


class BaseACO(NetworkxTSP):
    """
    Ant system optimization algorithm for the traveling-salesman problem. Ants act as
    agents that construct tours. Tours are guided by pheromone trails.

    Initially, m ants are placed on random cities. At each city, a state transition rule
    is applied. Trails that are close and with high pheromone strength are
    probabilistically preferred. Each ant has a tabu list which stores the partial tour.

    Once all ants have completed a tour, the pheromones are updated. This is done by
    lowering the pheromone trail strengths by a constant and then allowing the ants to
    deposit pheromone on the arcs they have visited. Arcs contained in shorter tours and
    that have been visited by many ants receive a higher pheromone update.
    """

    def __init__(
        self,
        name,
        filepath,
        alpha,
        beta,
        rho,
        iterations,
    ):
        super().__init__(name, filepath)
        # Hyperparameter that amplifies pheromone trails. Generally 1 is seen as the
        #   best value, as anything greater results in stagnation.
        self.alpha = alpha
        # Hyperparameter that amplifies node distance. Generally chosen to be between
        #   2 and 5
        self.beta = beta
        # Cache heuristic information
        self.eta = np.zeros(shape=(self.n + 1, self.n + 1))
        # Factor used in reducing pheromone strength over time. Can also be used to
        #   influence the initial value of pheromone strength
        self.rho = rho
        # Used in initializing pheromone strength. Actual calculation for tau_0 differs
        #   across ACO algorithm implementations
        self.nn_cost, _ = NearestNeighborSearch(filepath).algorithm()
        # Tracks pheromone strength.
        self.tau = np.ndarray(shape=(self.n + 1, self.n + 1))
        # Initialize ants. Can start each ant at each city involved in the problem
        self.ants = []
        for node in list(self.G.nodes):
            ant = Ant(node)
            self.ants.append(ant)
        # How many times each ant generates a tour
        self.iterations = iterations
        self.initialize_tau()

    @property
    def m(self):
        """
        Number of ants. Generally chosen to be equal to the number of nodes in the
        problem

        Returns
        -------
        int
        """
        return self.n

    def initialize_tau(self):
        """
        Initializes pheromone strength for all paths. Too low can cause bias in early
        tours, and too high increases time to convergence

        Returns
        -------
        None
        """
        self.tau[:] = self.n / self.nn_cost

    def update_trails(self, *args):
        """
        Performs pheromone evaporation and then update.

        Evaporation reduces pheromone strength by a factor of (1-rho). Allows bad tours
        to be forgotten over time.

        Update increases the pheromone strength of a node if the ant has traveled the
        connection during a tour. Update is influenced by quality of route.

        Should be performed after each run.

        Returns
        -------
        int
            Iter
        """
        for i in range(self.n):
            for j in range(self.n):
                # Evaporation
                self.tau[i, j] = (1 - self.rho) * self.tau[i, j]
                # Update
                for ant in self.ants:
                    if (i, j) in ant.arcs:
                        self.tau[i, j] += 1 / ant.cost

    def algorithm(self):
        costs = []
        for i in range(self.iterations):
            self.reset_ants()
            self.construct_solutions()
            self.update_trails(i)
            if config.debug:
                costs.append(min(ant.cost for ant in self.ants))
        if config.debug:
            plot_costs(costs)
        best_cost = np.inf
        best_route = None
        for ant in self.ants:
            if ant.cost < best_cost:
                best_cost = ant.cost
                best_route = ant.route
        return best_cost, np.array(best_route)

    def reset_ants(self):
        for ant in self.ants:
            ant.reset()

    def probabilistic_action_choice(self, i, j, N):
        tau_ij = self.tau[i, j]
        eta_ij = self.heuristic(i, j)
        numerator = tau_ij**self.alpha * eta_ij**self.beta
        denominator = 0
        for l in N:
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
        for ant in self.ants:
            source = ant.starting_node
            remaining_nodes = set(self.G.nodes) - set(ant.route)
            while remaining_nodes:
                best_p = 0
                best_dest = None
                for dest in remaining_nodes:
                    p_dest = self.probabilistic_action_choice(
                        source, dest, remaining_nodes
                    )
                    if p_dest > best_p:
                        best_p = p_dest
                        best_dest = dest
                self.add_arc(ant, source, best_dest)
                source = best_dest
                remaining_nodes = set(self.G.nodes) - set(ant.route)
            self.add_arc(ant, source, ant.starting_node)

    def add_arc(self, ant, i, j):
        ant.route.append(j)
        ant.arcs.add((i, j))
        ant.cost += self.dist(i, j)
