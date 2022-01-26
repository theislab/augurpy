from matplotlib import pyplot as plt


def plot_lollipop(results: dict):
    """Plot a lollipop plot of the mean augur values."""
    # using subplots() to draw vertical lines
    fig, axes = plt.subplots()
    y_axes_range = range(1, len(results["summary_metrics"].columns) + 1)
    axes.hlines(
        y_axes_range,
        xmin=0,
        xmax=results["summary_metrics"].sort_values("mean_augur_score", axis=1).loc["mean_augur_score"],
    )

    # drawing the markers (circle)
    axes.plot(
        results["summary_metrics"].sort_values("mean_augur_score", axis=1).loc["mean_augur_score"], y_axes_range, "o"
    )

    # formatting and details
    plt.xlabel("Mean Augur Score")
    plt.ylabel("Cell Type")
    plt.title("Augur Scores")
    plt.yticks(y_axes_range, results["summary_metrics"].sort_values("mean_augur_score", axis=1).columns)
