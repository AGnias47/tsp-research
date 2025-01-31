class Ant:
    def __init__(self):
        # visited nodes, also used to retrace the tour
        self.tabu_list = []