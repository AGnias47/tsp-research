class Ant:
    """
    Acts as the Agent for Ant Colony Optimization algorithms.

    Attributes
    ----------
    starting_node: int
        Origin node of the route traveled by the ant
    route: list
        Initialized with the starting node
    arcs: set
        Arcs of the graph that have been traveled. For example, if the ant goes from
        node 1 to 3, (1,3) will be added to the set
    cost: int
        Cost of the route found by the ant
    """

    def __init__(self, starting_node):
        self.starting_node = starting_node
        self.route = [starting_node]  # visited nodes, also used as a tabu list
        self.arcs = set()
        self.cost = 0

    def reset(self) -> None:
        """
        Sets route and costs back to origin. Should be performed before each iteration.

        Returns
        -------
        None
        """
        self.route = [self.starting_node]
        self.cost = 0
