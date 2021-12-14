from typing import Any, Dict, Tuple, Union

from anndata import AnnData
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, r2_score, roc_auc_score
from sklearn.model_selection import cross_validate


def set_scorer(
    estimator: Union[RandomForestRegressor, RandomForestClassifier, LogisticRegression],
) -> Tuple[Dict[str, Any], str]:
    """Set scoring fuctions for cross-validation based on estimator.

    Args:
        estimator: classifier object used to fit the model used to calculate the area under the curve

    Returns:
        Dict linking name to scorer object and string name
    """
    return (
        ({"auc": make_scorer(roc_auc_score)}, "auc")
        if isinstance(estimator, RandomForestClassifier) or isinstance(estimator, LogisticRegression)
        else ({"r2": make_scorer(r2_score)}, "r2")
    )


def run_cross_validation(
    subsample: AnnData,
    estimator: Union[RandomForestRegressor, RandomForestClassifier, LogisticRegression],
    subsample_idx: int,
    folds: int = 3,
) -> Dict:
    """Perform cross validation on given subsample.

    Args:
        subsample: subsample of gene expression matrix of size subsample_size
        estimator: classifier object to use in calculating the area under the curve
        folds: number of folds
        subsample_idx: index of subsample

    Returns:
        Dictionary containing prediction metrics and estimator for each fold.
    """
    scorer, name = set_scorer(estimator)
    x = subsample.to_df()
    y = subsample.obs[[col for col in subsample.obs if col.startswith("y")]]

    results = cross_validate(
        estimator=estimator,
        X=x,
        y=y.values.ravel(),
        scoring=scorer,
        cv=folds,
        return_estimator=True,
    )

    results["subsample_idx"] = subsample_idx
    results["mean_score"] = results[f"test_{name}"].mean()

    return results
