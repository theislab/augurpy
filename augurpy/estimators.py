from typing import Dict, Literal, Union

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from typing_extensions import TypedDict


class Params(TypedDict, total=False):
    """Type signature for random forest and logistic regression parameters."""

    n_estimators: int
    max_depth: int
    max_features: Union[Literal["auto"], Literal["log2"], Literal["sqrt"], int, float]
    penalty: Union[Literal["l1"], Literal["l2"], Literal["elasticnet"], Literal["none"]]


def create_estimator(
    classifier: Union[Literal["rf"], Literal["lr"]],
    params: Params,
    classification: bool = True,
) -> Union[RandomForestClassifier, RandomForestRegressor, LogisticRegression]:
    """Estimator object.

    Args:
        classifier: classifier to use in calculating the area under the curve either random forest or logistic regression
        params: list of parameters for estimator

    Returns:
        Estimator object that classifies.
    """
    if classifier == "rf" and classification:
        estimator = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            max_features=params.get("max_features", "auto"),
        )
        print(estimator, " Object created.")

    elif classifier == "rf" and not classification:
        estimator = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            max_features=params.get("max_features", "auto"),
        )
        print(estimator, " Object created.")

    elif classifier == "lf" and not classification:
        estimator = LogisticRegression()
        print(estimator + " Object created.")

    else:
        raise Exception("Missing valid input.")

    return estimator


def get_feature_importances(
    estimator: Union[RandomForestClassifier, RandomForestRegressor, LogisticRegression]
) -> Dict:
    """Get feature importances with respect to this estimator.

    Args:
        estimator: fitted classifier used to calculate feature importances

    Returns:
        Dictionary containing the importance of each feature
    """
    pass
