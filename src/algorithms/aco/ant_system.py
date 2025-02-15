from config import config
from src.algorithms.aco.base_aco import BaseACO


class AntSystem(BaseACO):
    algorithm_name = "Ant System ACO"
    abbreviation = "as"

    def __init__(self, filepath: str):
        super().__init__(
            filepath=filepath,
            alpha=config.ant_system["alpha"],
            beta=config.ant_system["beta"],
            rho=config.ant_system["rho"],
            iterations=config.ant_system["iterations"],
        )

    @property
    def big_o_runtime(self) -> int:
        """
        Rough estimate. Haven't done a deep dive into what this should actually be.

        Returns
        -------
        int
        """
        return config.ant_system["iterations"] * self.m * self.n**2
