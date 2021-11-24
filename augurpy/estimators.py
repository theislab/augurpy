from typing import Dict, List, Optional, Union

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression


def create_estimator(
    classifier: Union[RandomForestClassifier, RandomForestRegressor, LogisticRegression],
    rf_params: Optional[List],
    lr_params: Optional[List],
) -> Union[RandomForestClassifier, RandomForestRegressor, LogisticRegression]:
    """Estimator object.

    Args:
        classifier: classifier to use in calculating the area under the curve either random forest or logistic regression
        rf_params: list of parameters for random forest
        lr_params: list of parameters for logistic regression

    Returns:
        Estimator object that classifies.

    """


def get_feature_importances(
    estimator: Union[RandomForestClassifier, RandomForestRegressor, LogisticRegression]
) -> Dict:
    """Get feature importances with respect to this estimator.

    Args:
        estimator: fitted classifier used to calculate importances

    Returns:
        Dictionary containing the importance of each feature

    """
