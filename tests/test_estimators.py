from augurpy.estimators import create_estimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def test_creation():
	assert isinstance(create_estimator('random_forest_regressor'), RandomForestRegressor)
