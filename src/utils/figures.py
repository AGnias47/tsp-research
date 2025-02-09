import matplotlib.pyplot as plt


def plot_costs(costs):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    ax.set_title("Cost over each iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.plot(costs)
    plt.show()
