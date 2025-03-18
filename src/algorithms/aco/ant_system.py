from config import config
from src.algorithms.aco.base_aco import BaseACO


class AntSystem(BaseACO):
    algorithm_name = "Ant System ACO"
    abbreviation = "as"

    def __init__(
        self,
        filepath: str,
        alpha: int = config.ant_system["alpha"],
        beta: int = config.ant_system["beta"],
        rho: float = config.ant_system["rho"],
        iterations: int = config.ant_system["iterations"],
    ):
        super().__init__(
            filepath=filepath,
            alpha=alpha,
            beta=beta,
            rho=rho,
            iterations=iterations,
        )

    @property
    def hyperparameters(self):
        return config.ant_system

    @property
    def big_o_runtime(self) -> int:
        """
        Rough estimate. Haven't done a deep dive into what this should actually be.

        Returns
        -------
        int
        """
        return config.ant_system["iterations"] * self.m * self.n**2
