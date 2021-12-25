from typing import Any, Dict, Optional, Union

from anndata import AnnData
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, r2_score, roc_auc_score
from sklearn.model_selection import KFold, cross_validate


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
        else {"augur_score": make_scorer(r2_score), "r2": make_scorer(r2_score)}
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
        random_state: set random seed

    Returns:
        Dictionary containing prediction metrics and estimator for each fold.
    """
    scorer = set_scorer(estimator)
    x = subsample.to_df()
    y = subsample.obs[[col for col in subsample.obs if col.startswith("y")]]
    cv = KFold(n_splits=folds, random_state=random_state, shuffle=True)

    results = cross_validate(
        estimator=estimator,
        X=x,
        y=y.values.ravel(),
        scoring=scorer,
        cv=cv,
        return_estimator=True,
    )

    results["subsample_idx"] = subsample_idx
    for score in scorer.keys():
        results[f"mean_{s}"] = results[f"test_{s}"].mean()

    return results
