from matplotlib import pyplot as plt


def plot_scatterplot(results1, results2, top_n=None):
    """Create scatterplot of two augur results."""
    cell_types = results1["summary_metrics"].columns

    fig, axes = plt.subplots()
    plt.scatter(
        results1["summary_metrics"].loc["mean_augur_score", cell_types],
        results2["summary_metrics"].loc["mean_augur_score", cell_types],
    )

    # add diagonal
    limits = max(axes.get_xlim(), axes.get_ylim())
    (diag_line,) = axes.plot(limits, limits, ls="--", c=".3")
    # formatting and details
    plt.xlabel("Augur scores 1")
    plt.ylabel("Augur scores 2")
    plt.title("Augur Scores")
