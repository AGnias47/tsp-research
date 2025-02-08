"""
Taken from 3.3.4 of Dorigo and Stützle - Ant Colony Optimization
Improves upon Ant System with 4 modifications:
- Allows only the best-so-far ant to deposit pheromone
- This can lead to stagnation, which is counteracted by limiting the possible range of
pheromone trail values
- Pheromone trails are initialized to the upper pheromone trail limit
- Pheromone trails are reinitialized each time the system approaches stagnation or when
no improved tour has been generated for a certain number of consecutive iterations.

References
----------
- https://iridia.ulb.ac.be/~mdorigo/ACO/aco-code/public-software.html - C-code from
T. Stützle. Used for assistance in determining self.tau_min
"""

import numpy as np

from config import config
from src.algorithms.aco_base import ACOBase
from src.models.ant import Ant


class MaxMinAntSystem(ACOBase):
    def __init__(self, filepath):
        super().__init__(
            name="Max-Min Ant System ACO",
            filepath=filepath,
            alpha=config.mmas["alpha"],
            beta=config.mmas["beta"],
            rho=config.mmas["rho"],
            iterations=config.mmas["iterations"],
        )
        self.best_so_far = Ant(0)
        self.best_so_far.cost = np.inf
        self.tau_max = np.inf
        self.tau_min = 0

    def initialize_tau(self):
        self.tau[:] = self.tau_max

    def calculate_tau_max(self):
        return 1 / (self.rho * self.best_so_far.cost)

    def calculate_tau_min(self):
        """
        Calculates tau min

        References
        ----------
        acotsp.c

        Returns
        -------

        """
        p_x = np.pow(0.05, 1 / self.n)
        return self.tau_max * ((1 - p_x) / ((nn_ants - 1) * p_x))

    def update_trails(self):
        """
        For the MMAS update, the best ant out of all iterations so far is determined,
        and only the routes for that ant are updated.

        This should be alternated with
        the best ant in just the iterations.

        Should be performed after each run.

        Returns
        -------
        None
        """
        for ant in self.ants:
            if ant.cost < self.best_so_far.cost:
                self.best_so_far = ant
        self.tau_max = self.calculate_tau_max()
        self.tau_min = self.calculate_tau_min()
        for i in range(self.n):
            for j in range(self.n):
                # Evaporation
                self.tau[i, j] = (1 - self.rho) * self.tau[i, j]
                # Update
                if (i, j) in self.best_so_far.arcs:
                    self.tau[i, j] += 1 / self.best_so_far.cost
