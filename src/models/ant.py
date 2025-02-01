class Ant:
    def __init__(self):
        # visited nodes, also used as a tabu list
        self.route = []
        self.arcs = set()
        self.cost = 0

    def start(self, starting_node):
        self.route.append(starting_node)

    def clear(self):
        self.route = []
        self.cost = 0

    def add_arc(self, i, j, cost):
        self.route.append(j)
        self.arcs.add((i,j))
        self.cost += cost
