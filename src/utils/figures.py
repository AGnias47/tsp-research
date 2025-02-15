"""
Module for displaying metrics and results.
"""

import matplotlib.pyplot as plt


def plot_costs(costs: list[int]) -> None:
    """
    Plots a list of route cost, where each entry in the costs list is considered the
    cost per iteration during training / tuning of a TSP algorithm.

    Effects
    -------
    Displays a line plot of the costs

    Parameters
    ----------
    costs: list

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    ax.set_title("Cost over each iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.plot(costs)
    plt.show()
