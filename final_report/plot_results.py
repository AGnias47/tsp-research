"""
Generating plots for the final report. Some disclaimers about this specific code
- No attempts are being made here to make this code more efficient.
- AI coding tools such as Google Gemini and GitHub Copilot were utilized

https://github.com/AGnias47/russo-ukranian-tweet-classification/blob/main/sentiment_analysis/results.py
https://stackoverflow.com/questions/33958068/matplotlib-how-to-plot-a-line-with-categorical-data-on-the-x-axis
https://stackoverflow.com/a/10035974/8728749

"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


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
    ax[0, 0].axis("off")
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 1].imshow(mpimg.imread("final_report/problem_images/berlin52.png"))
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 2].imshow(mpimg.imread("final_report/problem_images/a280.png"))
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
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
    ax[1, 0].set_ylabel("Cost")
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
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 1].imshow(mpimg.imread("final_report/problem_images/asterisk100.png"))
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
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
    ax[1, 0].set_ylabel("Cost")
    fig.suptitle("Star Problem Images with Graphical Cost Results")
    plt.savefig("final_report/png/star.png")
    plt.show()


def plot_spiral():
    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 3), sharex=False, sharey=False
    )
    ax[0].imshow(mpimg.imread("final_report/problem_images/spiral79.png"))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].plot("spiral79", 286, "o", color="magenta", label="concorde")
    ax[1].plot("spiral79", 298, "o", color=[1.0, 0.45, 0.15], label="nns")
    ax[1].plot("spiral79", 292, "o", color=[1.0, 0, 0], label="mmas")
    ax[1].plot("spiral79", 301, "o", color="purple", label="dq")
    ax[1].plot("spiral79", 306, "o", label="dqn")
    ax[1].grid(True)
    ax[1].set_ylabel("Cost")
    fig.legend(*ax[1].get_legend_handles_labels(), loc="lower right")
    fig.suptitle("Spiral Problem Image with Graphical Cost Results")
    plt.savefig("final_report/png/spiral.png")
    plt.show()


def plot_cluster():
    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 3), sharex=False, sharey=False
    )
    ax[0].imshow(mpimg.imread("final_report/problem_images/5cluster43.png"))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
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


def plot_tetrahedral():
    problems_symmetric = ["Tnm52", "Tnm70", "Tnm160"]
    concorde = [551609, 881036, None]
    mmas = [607236, 994978, 2623096]
    nns = [632300, 1005243, 2694609]
    dq = [672125, 1032075, 2767928]
    fig, ax = plt.subplots(
        nrows=2, ncols=3, figsize=(13, 6), sharex=False, sharey=False
    )
    ax[0, 0].imshow(mpimg.imread("final_report/problem_images/Tnm52.png"))
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 1].imshow(mpimg.imread("final_report/problem_images/Tnm70.png"))
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 2].imshow(mpimg.imread("final_report/problem_images/Tnm160.png"))
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    for i in range(len(problems_symmetric)):
        try:
            ax[1, i].plot(
                problems_symmetric[i],
                concorde[i],
                "o",
                color="magenta",
                label="concorde",
            )
        except ValueError:
            pass
        ax[1, i].plot(
            problems_symmetric[i], nns[i], "o", color=[1.0, 0.45, 0.15], label="nns"
        )
        ax[1, i].plot(
            problems_symmetric[i], mmas[i], "o", color=[1.0, 0, 0], label="mmas"
        )
        ax[1, i].plot(problems_symmetric[i], dq[i], "o", color="purple", label="dq")
        ax[1, i].grid(True)
    fig.legend(*ax[1, 0].get_legend_handles_labels(), loc="lower right")
    ax[1, 0].set_ylabel("Cost")
    fig.suptitle("Tetrahedral Problem Images with Graphical Cost Results")
    plt.savefig("final_report/png/tetrahedral.png")
    plt.show()


def plot_asymmetrical_small_medium():
    problems_symmetric = ["br17", "ftv47"]
    optimum = [39, 1776]
    mmas = [82, 2173]
    nns = [92, 2374]
    dq = [92, 2374]
    dqn = [92, 3046]
    rc = [70, 5225]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 2), sharex=False, sharey=False)
    for i in range(len(problems_symmetric)):
        ax[i].plot(
            problems_symmetric[i], optimum[i], "o", color="magenta", label="hk/optim"
        )
        ax[i].plot(
            problems_symmetric[i], nns[i], "o", color=[1.0, 0.45, 0.15], label="nns"
        )
        ax[i].plot(problems_symmetric[i], mmas[i], "o", color=[1.0, 0, 0], label="mmas")
        ax[i].plot(problems_symmetric[i], dq[i], "o", color="purple", label="dq")
        ax[i].plot(problems_symmetric[i], dqn[i], "o", label="dqn")
        ax[i].plot(problems_symmetric[i], rc[i], "o", color="brown", label="rc")
        ax[i].grid(True)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower right")
    ax[0].set_ylabel("Cost")
    fig.suptitle("Asymmetrical Small-Medium Problems Graphical Cost Results")
    plt.savefig("final_report/png/asymmetric_small-medium.png")
    plt.show()


def plot_asymmetrical_large():
    problems_symmetric = ["rbg323", "rbg358", "rbg403"]
    optimum = [1326, 1163, 2465]
    nns = [1734, 1812, 3535]
    mmas = [3488, 4180, 5685]
    dq = [2245, 1817, 3457]
    dqn = [6119, 6711, 7575]
    rc = [5695, 6378, 7187]
    fig, ax = plt.subplots(
        nrows=1, ncols=3, figsize=(10, 2), sharex=False, sharey=False
    )
    for i in range(len(problems_symmetric)):
        ax[i].plot(
            problems_symmetric[i], optimum[i], "o", color="magenta", label="optim"
        )
        ax[i].plot(
            problems_symmetric[i], nns[i], "o", color=[1.0, 0.45, 0.15], label="nns"
        )
        ax[i].plot(problems_symmetric[i], mmas[i], "o", color=[1.0, 0, 0], label="mmas")
        ax[i].plot(problems_symmetric[i], dq[i], "o", color="purple", label="dq")
        ax[i].plot(problems_symmetric[i], dqn[i], "o", label="dqn")
        ax[i].plot(problems_symmetric[i], rc[i], "o", color="brown", label="rc")
        ax[i].grid(True)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower right")
    ax[0].set_ylabel("Cost")
    fig.suptitle("Asymmetrical Large Problems Graphical Cost Results")
    plt.savefig("final_report/png/asymmetric_large.png")
    plt.show()


def spiral_solutions():
    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 3), sharex=False, sharey=False
    )
    ax[0].imshow(mpimg.imread("final_report/solutions/spiralsolved.png"))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlabel("Concorde")
    ax[1].imshow(mpimg.imread("final_report/solutions/spiral_nn.png"))
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xlabel("NNS")
    fig.legend(*ax[1].get_legend_handles_labels(), loc="lower right")
    plt.savefig("final_report/png/spiral_concorde_nn.png")
    plt.show()


def cluster_solutions():
    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 3), sharex=False, sharey=False
    )
    ax[0].imshow(mpimg.imread("final_report/solutions/clustersconcorde.png"))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlabel("Concorde")
    ax[1].imshow(mpimg.imread("final_report/solutions/clusters_nn.png"))
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xlabel("NNS")
    fig.legend(*ax[1].get_legend_handles_labels(), loc="lower right")
    plt.savefig("final_report/png/cluster.png")
    plt.show()


if __name__ == "__main__":
    #plot_symmetric()
    #plot_star()
    #plot_spiral()
    plot_cluster()
    #plot_tetrahedral()
    #plot_asymmetrical_small_medium()
    #plot_asymmetrical_large()
    # spiral_solutions()
