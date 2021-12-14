import scanpy as sc

from augurpy.cv import run_cross_validation
from augurpy.estimators import create_estimator
from augurpy.read_load import load

adata = sc.read_h5ad("tests/sc_sim.h5ad")
adata = load(adata)

# subsample 100 obs
sc.pp.subsample(adata, n_obs=100)

rf_estimator = create_estimator("random_forest_classifier")


def test_run_cross_validation():
    """Test run cross validation with categorical and classifier."""
    cv = run_cross_validation(adata, rf_estimator, subsample_idx=1, folds=3)
    assert cv["test_auc"] is not None
