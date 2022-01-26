from matplotlib import pyplot as plt


def plot_differential_prioritization(results, top_n=None):
    """Plot result of differential prioritization."""
    x = results["mean_augur_score1"]
    y = results["mean_augur_score2"]

    fig, axes = plt.subplots()
    scatter = plt.scatter(x, y, c=results.z, cmap="Greens")

    # add diagonal
    limits = max(axes.get_xlim(), axes.get_ylim())
    (diag_line,) = axes.plot(limits, limits, ls="--", c=".3")

    # formatting and details
    plt.xlabel("Augur scores 1")
    plt.ylabel("Augur scores 2")
    plt.title("Differential Prioritization")
    legend1 = axes.legend(*scatter.legend_elements(), loc="center left", title="z-scores", bbox_to_anchor=(1, 0.5))
    axes.add_artist(legend1)
