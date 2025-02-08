"""
Q-Learning Reinforcement Algorithm for solving the Traveling-Salesman problem
"""

from src.models.networkx_tsp import NetworkxTSP


class QLearning(NetworkxTSP):
    def __init__(self, filepath):
        super().__init__("Q-Learning", filepath)

    @property
    def big_o_runtime(self):
        return None
