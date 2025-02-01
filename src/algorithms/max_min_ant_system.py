"""
Taken from 3.3.4 of Dorigo and Stützle - Ant Colony Optimization
Improves upon Ant System with 4 modifications:
- Allows only the iteration-best ant or the best-so-far ant to deposit pheromone
- This can lead to stagnation, which is counteracted by limiting the possible range of
pheromone trail values
- Pheromone trails are initialized to the upper pheromone trail limit
- Pheromone trails are reinitialized each time the system approaches stagnation or when
no improved tour has been generated for a certain number of consecutive iterations.
"""

from config import config
from src.algorithms.aco_base import ACOBase


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
