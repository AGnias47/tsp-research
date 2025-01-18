from utils.decorators import timing


class TSP:
    @timing
    def run_tsp(self):
        return self.run()

    def run(self):
        raise NotImplementedError
