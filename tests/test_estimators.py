from sklearn.ensemble import RandomForestRegressor

from augurpy.estimators import create_estimator


def test_creation():
    """Test output of create_estimator."""
    assert isinstance(create_estimator("random_forest_regressor"), RandomForestRegressor)
