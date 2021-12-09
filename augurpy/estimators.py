from dataclasses import dataclass
from typing import Dict, Literal, Union

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from switchlang import switch
from typing_extensions import TypedDict


@dataclass
class Params(TypedDict, total=False):
    """Type signature for random forest and logistic regression parameters."""

    n_estimators: int
    max_depth: int
    max_features: Union[Literal["auto"], Literal["log2"], Literal["sqrt"], int, float]
    penalty: Union[Literal["l1"], Literal["l2"], Literal["elasticnet"], Literal["none"]]


def _raise_exception(exception_message: str):
    """Raise exception for invalid classifier input."""
    raise Exception(exception_message)


def create_estimator(
    classifier: Union[
        Literal["random_forest_classifier"],
        Literal["random_forest_regressor"],
        Literal["logistic_regression_classifier"],
    ],
    params: Params,
) -> Union[RandomForestClassifier, RandomForestRegressor, LogisticRegression]:
    """Creates a model object of the provided type and populates it with desired parameters.

    Args:
        classifier: classifier to use in calculating the area under the curve.
                    Either random forest classifier or logistic regression for categorical data
                    or random forest regressor for continous data
        params: parameters used to populate the model object.
                n_estimators defines the number of trees in the forest;
                max_depth specifies the maximal depth of each tree;
                max_features specifies the maximal number of features considered when looking at best split,
                    if int then consider max_features for each split
                    if float consider round(max_features*n_features)
                    if `auto` then max_features=n_features (default)
                    if `log2` then max_features=log2(n_features)
                    if `sqrt` then max_featuers=sqrt(n_features)
                penalty defines the norm of the penalty used in logistic regression
                    if `l1` then L1 penalty is added
                    if `l2` then L2 penalty is added (default)
                    if `elasticnet` both L1 and L2 penalties are added
                    if `none` no penalty is added

    Returns:
        Estimator object.
    """
    with switch(classifier) as c:
        c.case(
            "random_forest_classifier",
            lambda: RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                max_features=params.get("max_features", "auto"),
            ),
        )
        c.case(
            "random_forest_regressor",
            lambda: RandomForestRegressor(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                max_features=params.get("max_features", "auto"),
            ),
        )
        c.case("logistic_regression_classifier", lambda: LogisticRegression(penalty=params.get("penalty", "l2")))
        c.default(_raise_exception)(
            "Missing valid input. Choose rf or lr for categorical labels and rf for continuous labels."
        )

    return c.result


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
