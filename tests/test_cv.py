from math import isclose

import scanpy as sc

from augurpy.cv import run_cross_validation
from augurpy.estimator import Params, create_estimator
from augurpy.read_load import load

adata = sc.read_h5ad("tests/sc_sim.h5ad")
adata = load(adata)

# subsample 100 obs
sc.pp.subsample(adata, n_obs=100, random_state=42)

rf_classifier = create_estimator("random_forest_classifier", Params(random_state=42))
lr_classifier = create_estimator("logistic_regression_classifier", Params(random_state=42))
rf_regressor = create_estimator("random_forest_regressor", Params(random_state=42))


def test_classifier():
    """Test run cross validation with classifier."""
    cv = run_cross_validation(adata, rf_classifier, subsample_idx=1, folds=3, random_state=42)
    auc = 0.802411
    assert any([isclose(cv["mean_auc"], auc, abs_tol=10 ** -5)])

    cv = run_cross_validation(adata, lr_classifier, subsample_idx=1, folds=3, random_state=42)
    auc = 0.869871
    assert any([isclose(cv["mean_auc"], auc, abs_tol=10 ** -5)])


def test_regressor():
    """Test run cross validation with regressor."""
    cv = run_cross_validation(adata, rf_regressor, subsample_idx=1, folds=3, random_state=42)
    ccc = 0.433445
    r2 = 0.274051
    assert any([isclose(cv["mean_ccc"], ccc, abs_tol=10 ** -5), isclose(cv["mean_r2"], r2, abs_tol=10 ** -5)])
