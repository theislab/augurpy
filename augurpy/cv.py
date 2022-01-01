"""Cross validate subsample to return results object containing metric scores and feature importances."""
from collections import defaultdict
from typing import Any, Dict, Optional, Union

import numpy as np
from anndata import AnnData
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, r2_score, roc_auc_score
from sklearn.model_selection import KFold, cross_validate


def ccc_score(y_true, y_pred) -> float:
    """Implementation of Lin's Concordance correlation coefficient, based on https://gitlab.com/-/snippets/1730605.

    Args:
        y_true: array-like of shape (n_samples), ground truth (correct) target values
        y_pred: array-like of shape (n_samples), estimated target values

    Returns:
        Concordance correlation coefficient.
    """
    # covariance between y_true and y_pred
    s_xy = np.cov([y_true, y_pred])[0, 1]
    # means
    x_m = np.mean(y_true)
    y_m = np.mean(y_pred)
    # variances
    s_x_sq = np.var(y_true)
    s_y_sq = np.var(y_pred)

    # condordance correlation coefficient
    ccc = (2.0 * s_xy) / (s_x_sq + s_y_sq + (x_m - y_m) ** 2)

    return ccc


def set_scorer(
    estimator: Union[RandomForestRegressor, RandomForestClassifier, LogisticRegression],
) -> Dict[str, Any]:
    """Set scoring fuctions for cross-validation based on estimator.

    Args:
        estimator: classifier object used to fit the model used to calculate the area under the curve

    Returns:
        Dict linking name to scorer object and string name
    """
    return (
        {"augur_score": make_scorer(roc_auc_score), "auc": make_scorer(roc_auc_score)}
        if isinstance(estimator, RandomForestClassifier) or isinstance(estimator, LogisticRegression)
        else {"augur_score": make_scorer(ccc_score), "r2": make_scorer(r2_score), "ccc": make_scorer(ccc_score)}
    )


def run_cross_validation(
    subsample: AnnData,
    estimator: Union[RandomForestRegressor, RandomForestClassifier, LogisticRegression],
    subsample_idx: int,
    folds: int = 3,
    random_state=Optional[int],
) -> Dict:
    """Perform cross validation on given subsample.

    Args:
        subsample: subsample of gene expression matrix of size subsample_size
        estimator: classifier object to use in calculating the area under the curve
        subsample_idx: index of subsample
        folds: number of folds
        random_state: set random fold seed

    Returns:
        Dictionary containing prediction metrics and estimator for each fold.
    """
    scorer = set_scorer(estimator)
    x = subsample.to_df()
    y = subsample.obs[[col for col in subsample.obs if col.startswith("y")]]
    folds = KFold(n_splits=folds, random_state=random_state, shuffle=True)

    results = cross_validate(
        estimator=estimator,
        X=x,
        y=y.values.ravel(),
        scoring=scorer,
        cv=folds,
        return_estimator=True,
    )

    results["subsample_idx"] = subsample_idx
    for score in scorer.keys():
        results[f"mean_{score}"] = results[f"test_{score}"].mean()

    # feature importances
    feature_importances = defaultdict(list)
    if isinstance(estimator, RandomForestClassifier) or isinstance(estimator, RandomForestRegressor):
        for fold, estimator in list(zip(range(len(results["estimator"])), results["estimator"])):
            feature_importances["genes"].extend(x.columns.tolist())
            feature_importances["feature_importances"].extend(estimator.feature_importances_.tolist())
            feature_importances["subsample_idx"].extend(len(x.columns) * [subsample_idx])
            feature_importances["fold"].extend(len(x.columns) * [fold])

    if isinstance(estimator, LogisticRegression):
        pass

    results["feature_importances"] = feature_importances

    return results
