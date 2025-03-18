"""
Taken from 3.3.4 of Dorigo and Stützle - Ant Colony Optimization.

Improves upon Ant System with 4 modifications:
- Allow only the best-so-far or iteration-best ant to deposit pheromone
- Limit the possible range of pheromone trail values
- Pheromone trails are initialized to the upper pheromone trail limit
- Pheromone trails are reinitialized each time the system approaches stagnation or when
  no improved tour has been generated for a certain number of consecutive iterations.

References
----------
* https://iridia.ulb.ac.be/~mdorigo/ACO/aco-code/public-software.html - C-code from
T. Stützle. Used for assistance in determining self.tau_min
* https://stackoverflow.com/a/5996949/8728749 - Efficiently limiting a value to a range
"""

import numpy as np

from config import config
from src.algorithms.aco.base_aco import BaseACO
from src.models.ant import Ant


class MaxMinAntSystem(BaseACO):
    algorithm_name = "Min-Max Ant System ACO"
    abbreviation = "mmas"

    def __init__(
        self,
        filepath: str,
        alpha: int = config.mmas["alpha"],
        beta: int = config.mmas["beta"],
        rho: float = config.mmas["rho"],
        iterations: int = config.mmas["iterations"],
        stagnation_tolerance: int = config.mmas["stagnation_tolerance"],
    ):
        super().__init__(
            filepath=filepath,
            alpha=alpha,
            beta=beta,
            rho=rho,
            iterations=iterations,
        )
        self.stagnation_tolerance = stagnation_tolerance
        self.best_so_far = Ant(0)
        self.best_so_far.cost = np.inf
        self.tau_max = np.inf
        self.tau_min = 0
        self.update_iteration = 0

    @property
    def hyperparameters(self):
        return config.mmas

    @property
    def big_o_runtime(self) -> int:
        """
        Rough estimate. Haven't done a deep dive into what this should actually be.
        Takes longer than Ant System and currently both of these functions are the same.

        Returns
        -------
        int
        """
        return config.mmas["iterations"] * self.m * self.n**2

    def initialize_tau(self) -> None:
        self.tau[:] = 1 / (self.rho * self.nn_cost)

    def calculate_tau_max(self) -> float:
        """
        Calculates tau max using the best-so-far cost

        References
        ----------
        ACOTSP.V1.03.tgz at
        https://iridia.ulb.ac.be/~mdorigo/ACO/aco-code/public-software.html - referenced
        acotsp.c

        Returns
        -------
        float
        """
        return 1 / (self.rho * self.best_so_far.cost)

    def calculate_tau_min(self) -> float:
        """
        Calculates tau min using tau max

        References
        ----------
        ACOTSP.V1.03.tgz at
        https://iridia.ulb.ac.be/~mdorigo/ACO/aco-code/public-software.html - referenced
        acotsp.c

        Returns
        -------
        float
        """
        p_x = np.pow(0.05, 1 / self.n)
        avg_cities = self.n / 2
        return self.tau_max * ((1 - p_x) / ((avg_cities - 1) * p_x))

    def update_trails(self, iteration: int) -> None:
        """
        For the MMAS update, only update paths for either the best-so-far ant, or the
        iteration-best ant. In this algorithm, an iteration count is tracked, and the
        best-so-far and iteration-best ants are alternated through for the update. The
        iteration-best ant encourages exploration, while the best-so-far ant is most
        likely to give the best update.

        If stagnation is detected, the pheromone matrix is reset. Stagnation is
        considered a number of consecutive iterations without improvement. This number
        is defined in the config as stagnation_tolerance.

        Parameters
        ----------
        iteration: int
            Iteration count, tracked by the algorithm

        References
        ----------
        Stagnation reset is modeled after ACOTSP/acotsp.c::search_control_and_statistics
        in ACOTSP.V1.03.tgz at
        https://iridia.ulb.ac.be/~mdorigo/ACO/aco-code/public-software.html

        Returns
        -------
        None
        """
        iteration_best_ant = self.ants[0]
        for ant in self.ants:
            if ant.cost < self.best_so_far.cost:
                self.best_so_far = ant
                self.update_iteration = iteration
            if ant.cost < iteration_best_ant.cost:
                iteration_best_ant = ant
        if (iteration - self.update_iteration) > self.stagnation_tolerance:
            # Stagnation reset procedure
            self.tau[:] = 1 / (self.rho * self.best_so_far.cost)
            self.update_iteration = iteration
            return
        if iteration % 2:
            ant_for_update = self.best_so_far
        else:
            ant_for_update = iteration_best_ant
        self.tau_max = self.calculate_tau_max()
        self.tau_min = self.calculate_tau_min()
        for i in range(self.n):
            for j in range(self.n):
                # Evaporation
                self.tau[i, j] = (1 - self.rho) * self.tau[i, j]
                # Update
                if (i, j) in ant_for_update.arcs:
                    self.tau[i, j] += max(
                        min(self.tau_min, 1 / ant_for_update.cost), self.tau_max
                    )
        return iteration_best_ant.cost
