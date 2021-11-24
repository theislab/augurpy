from typing import List, Union

from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression


def run_cross_validate(
    subsample: DataFrame,
    estimator: Union[RandomForestRegressor, RandomForestClassifier, LogisticRegression],
    metrics: List,
    folds: int,
    subsample_idx: int,
) -> List:
    """Perform cross validation on given subsample.

    Args:
        subsample: subsample of gene expression matrix of size subsample_size
        estimator: classifier object to use in calculating the area under the curve
        metrics: list of metrics to measure in each cv-fold
        folds: number of folds
        subsample_idx: index of subsample

    Returns:
        Data frame containing prediction metrics for each fold

    """
