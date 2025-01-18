from abc import ABC, abstractmethod

class TSP(ABC):
    @abstractmethod
    def run(self):
        pass