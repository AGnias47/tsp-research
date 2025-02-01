class Ant:
    def __init__(self, starting_node):
        self.starting_node = starting_node
        # visited nodes, also used as a tabu list
        self.route = [starting_node]
        self.arcs = set()
        self.cost = 0

    def reset(self):
        self.route = [self.starting_node]
        self.cost = 0

    def add_arc(self, i, j, cost):
        self.route.append(j)
        self.arcs.add((i, j))
        self.cost += cost
