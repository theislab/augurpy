<<<<<<< HEAD
from sklearn.ensemble import RandomForestRegressor

from augurpy.estimators import create_estimator
=======
import pytest
from sklearn.ensemble import RandomForestRegressor

from augurpy.estimators import Params, create_estimator
>>>>>>> development


def test_creation():
    """Test output of create_estimator."""
    assert isinstance(create_estimator("random_forest_regressor"), RandomForestRegressor)
<<<<<<< HEAD
=======


def test_missing_value():
    """Test raising missing value error."""
    with pytest.raises(Exception):
        create_estimator("this is no estimator")


def test_params():
    """Test parameters."""
    rf_estimator = create_estimator("random_forest_classifier", Params(n_estimators=9, max_depth=10, penalty=13))
    lr_estimator = create_estimator("logistic_regression_classifier", Params(penalty="elasticnet"))
    assert rf_estimator.get_params()["n_estimators"] == 9
    assert rf_estimator.get_params()["max_depth"] == 10
    assert lr_estimator.get_params()["penalty"] == "elasticnet"

    with pytest.raises(TypeError):
        create_estimator("random_forest_regressor", Params(unvalid=10))
>>>>>>> development
