import tsplib95
from networkx import Graph


def tsplib_graph(filename: str) -> Graph:
    """
    Given a file path to a tsplib formatted file, return a networkx.Graph object

    Parameters
    ----------
    filename: str

    Returns
    -------
    networkx.Graph
    """
    return tsplib95.load(filename).get_graph()
