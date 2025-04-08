"""
https://github.com/AGnias47/russo-ukranian-tweet-classification/blob/main/sentiment_analysis/results.py
https://stackoverflow.com/questions/33958068/matplotlib-how-to-plot-a-line-with-categorical-data-on-the-x-axis

No attempts are being made here to make this code more efficient.
"""


import matplotlib.pyplot as plt


def plot_symmetric():
    problems_symmetric = ["fri26", "berlin52", "a280"]
    concorde = [937, 7542, 2579]
    nns = [1112, 8980, 3157]
    mmas = [955, 8092, 2988]
    dq = [1050, 9179, 5526]
    dqn = [1184, 13462, 31082]
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=False, sharey=False)
    for i in range(len(problems_symmetric)):
        ax[i].plot(problems_symmetric[i], concorde[i], "o", color="magenta", label="concorde")
        ax[i].plot(problems_symmetric[i], nns[i], "o", color=[1.0, 0.45, 0.15], label="nns")
        ax[i].plot(problems_symmetric[i], mmas[i], "o", color=[1.0, 0, 0], label="mmas")
        ax[i].plot(problems_symmetric[i], dq[i], "o", color="purple", label="dq")
        ax[i].plot(problems_symmetric[i], dqn[i], "o", label="dqn")
        ax[i].grid(True)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower right")
    fig.supylabel("Cost")
    fig.suptitle("Results of the Symmetric Problem")
    plt.savefig("final_report/png/symmetric.png")
    plt.show()


def plot_asymmetric():
    problems_asymmetric = ["br17", "ftv47", "rbg403"]
    optimal = [39, 1776, 2465]
    nns = [92, 2374, 3535]
    mmas = [82, 2173, 5685]
    dq = []
    dqn = []
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=False, sharey=False)
    for i in range(len(problems_asymmetric)):
        ax[i].plot(problems_asymmetric[i], optimal[i], "o", color="magenta", label="optimal")
        ax[i].plot(problems_asymmetric[i], nns[i], "o", color=[1.0, 0.45, 0.15], label="nns")
        ax[i].plot(problems_asymmetric[i], mmas[i], "o", color=[1.0, 0, 0], label="mmas")
        ax[i].plot(problems_asymmetric[i], dq[i], "o", color="purple", label="dq")
        ax[i].plot(problems_asymmetric[i], dqn[i], "o", label="dqn")
        ax[i].grid(True)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower right")
    fig.supylabel("Cost")
    fig.suptitle("Results of the Symmetric Problem")
    plt.savefig("final_report/png/symmetric.png")
    plt.show()


if __name__ == "__main__":
    plot_symmetric()