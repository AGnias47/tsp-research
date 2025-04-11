"""
https://github.com/AGnias47/russo-ukranian-tweet-classification/blob/main/sentiment_analysis/results.py
https://stackoverflow.com/questions/33958068/matplotlib-how-to-plot-a-line-with-categorical-data-on-the-x-axis

No attempts are being made here to make this code more efficient.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_symmetric():
    problems_symmetric = ["fri26", "berlin52", "a280"]
    concorde = [937, 7542, 2579]
    nns = [1112, 8980, 3157]
    mmas = [955, 8092, 2988]
    dq = [1050, 9179, 3554]
    dqn = [1027, 10922, 7817]
    fig, ax = plt.subplots(
        nrows=2, ncols=3, figsize=(13, 6), sharex=False, sharey=False
    )
    # ax[0, 0].imshow(mpimg.imread("final_report/problem_images/berlin52.png"))
    ax[0, 1].imshow(mpimg.imread("final_report/problem_images/berlin52.png"))
    ax[0, 2].imshow(mpimg.imread("final_report/problem_images/a280.png"))
    for i in range(len(problems_symmetric)):
        ax[1, i].plot(
            problems_symmetric[i], concorde[i], "o", color="magenta", label="concorde"
        )
        ax[1, i].plot(
            problems_symmetric[i], nns[i], "o", color=[1.0, 0.45, 0.15], label="nns"
        )
        ax[1, i].plot(
            problems_symmetric[i], mmas[i], "o", color=[1.0, 0, 0], label="mmas"
        )
        ax[1, i].plot(problems_symmetric[i], dq[i], "o", color="purple", label="dq")
        ax[1, i].plot(problems_symmetric[i], dqn[i], "o", label="dqn")
        ax[1, i].grid(True)
    fig.legend(*ax[1, 0].get_legend_handles_labels(), loc="lower right")
    fig.supylabel("Cost                          Distance")
    fig.suptitle("Symmetric Problem Images with Graphical Cost Results")
    plt.savefig("final_report/png/symmetric.png")
    plt.show()


def plot_star():
    problems_symmetric = ["star85", "asterisk100"]
    concorde = [275, 254]
    mmas = [295, 270]
    nns = [358, 345]
    dq = [390, 373]
    dqn = [464, 553]
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 5), sharex=False, sharey=False)
    ax[0, 0].imshow(mpimg.imread("final_report/problem_images/star85.png"))
    ax[0, 1].imshow(mpimg.imread("final_report/problem_images/asterisk100.png"))
    for i in range(len(problems_symmetric)):
        ax[1, i].plot(
            problems_symmetric[i], concorde[i], "o", color="magenta", label="concorde"
        )
        ax[1, i].plot(
            problems_symmetric[i], nns[i], "o", color=[1.0, 0.45, 0.15], label="nns"
        )
        ax[1, i].plot(
            problems_symmetric[i], mmas[i], "o", color=[1.0, 0, 0], label="mmas"
        )
        ax[1, i].plot(problems_symmetric[i], dq[i], "o", color="purple", label="dq")
        ax[1, i].plot(problems_symmetric[i], dqn[i], "o", label="dqn")
        ax[1, i].grid(True)
    fig.legend(*ax[1, 0].get_legend_handles_labels(), loc="lower right")
    fig.supylabel("Cost                          Distance")
    fig.suptitle("Star Problem Images with Graphical Cost Results")
    plt.savefig("final_report/png/star.png")
    plt.show()


def plot_spiral():
    problems_symmetric = ["spiral79", "dh39"]
    concorde = [286, 123]
    mmas = [292, 140]
    nns = [298, 129]
    dq = [301, 128]
    dqn = [306, 155]
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 5), sharex=False, sharey=False)
    ax[0, 0].imshow(mpimg.imread("final_report/problem_images/spiral79.png"))
    # ax[0, 1].imshow(mpimg.imread("final_report/problem_images/dh39.png"))
    for i in range(len(problems_symmetric)):
        ax[1, i].plot(
            problems_symmetric[i], concorde[i], "o", color="magenta", label="concorde"
        )
        ax[1, i].plot(
            problems_symmetric[i], nns[i], "o", color=[1.0, 0.45, 0.15], label="nns"
        )
        ax[1, i].plot(
            problems_symmetric[i], mmas[i], "o", color=[1.0, 0, 0], label="mmas"
        )
        ax[1, i].plot(problems_symmetric[i], dq[i], "o", color="purple", label="dq")
        ax[1, i].plot(problems_symmetric[i], dqn[i], "o", label="dqn")
        ax[1, i].grid(True)
    fig.legend(*ax[1, 0].get_legend_handles_labels(), loc="lower right")
    fig.supylabel("Cost                          Distance")
    fig.suptitle("Spiral Problem Images with Graphical Cost Results")
    plt.savefig("final_report/png/spiral.png")
    plt.show()


def plot_cluster():
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3), sharex=False, sharey=False)
    ax[0].imshow(mpimg.imread("final_report/problem_images/5cluster43.png"))
    ax[0].set_ylabel("Distance")
    ax[1].plot("5cluster43", 100, "o", color="magenta", label="concorde")
    ax[1].plot("5cluster43", 111, "o", color=[1.0, 0.45, 0.15], label="nns")
    ax[1].plot("5cluster43", 112, "o", color=[1.0, 0, 0], label="mmas")
    ax[1].plot("5cluster43", 118, "o", color="purple", label="dq")
    ax[1].plot("5cluster43", 141, "o", label="dqn")
    ax[1].grid(True)
    ax[1].set_ylabel("Cost")
    fig.legend(*ax[1].get_legend_handles_labels(), loc="lower right")
    fig.suptitle("Cluster Problem Image with Graphical Cost Results")
    plt.savefig("final_report/png/cluster.png")
    plt.show()


if __name__ == "__main__":
    # plot_symmetric()
    # plot_star()
    # plot_spiral()
    plot_cluster()
