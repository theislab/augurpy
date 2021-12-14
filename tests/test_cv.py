import scanpy as sc
from math import isclose
from augurpy.cv import run_cross_validation
from augurpy.estimators import Params, create_estimator
from augurpy.read_load import load

adata = sc.read_h5ad("tests/sc_sim.h5ad")
adata = load(adata)

# subsample 100 obs
sc.pp.subsample(adata, n_obs=100, random_state=42)

rf_estimator = create_estimator("random_forest_classifier", Params(random_state=42))


def test_run_cross_validation():
    """Test run cross validation with categorical and classifier."""
    cv = run_cross_validation(adata, rf_estimator, subsample_idx=1, folds=3)
    auc = [0.8472222222222222, 0.7205882352941176, 0.6777777777777777]    
    assert any([isclose(x, y, abs_tol=10**-15)for x, y in list(zip(cv["test_auc"].tolist(), auc))])
