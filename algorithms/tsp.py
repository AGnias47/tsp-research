from utils.decorators import timing


class TSP:
    @timing
    def run_tsp(self):
        return self.algorithm()

    def algorithm(self):
        raise NotImplementedError
