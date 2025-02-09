from config import config
from src.algorithms.aco.base_aco import BaseACO


class AntSystem(BaseACO):
    """
    Baseline Ant Colony Optimization Algorithm
    """

    def __init__(self, filepath):
        super().__init__(
            name="Ant Colony Optimization",
            filepath=filepath,
            alpha=config.ant_system["alpha"],
            beta=config.ant_system["beta"],
            rho=config.ant_system["rho"],
            iterations=config.ant_system["iterations"],
        )

    @property
    def big_o_runtime(self) -> int:
        return config.ant_system["iterations"] * self.m * self.n**2
