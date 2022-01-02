<<<<<<< HEAD
=======
import os
>>>>>>> f0e9d4c (add pyupgrade)
from math import isclose

import numpy as np
import scanpy as sc

from augurpy.estimator import Params, create_estimator
from augurpy.evaluate import calculate_auc, run_cross_validation
from augurpy.read_load import load

CWD = os.getcwd()

sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
sc_sim_adata = load(sc_sim_adata)
estimator = create_estimator("random_forest_classifier", Params(random_state=42))


def test_calculate_auc():
    """Tests auc calculation."""
    adata, results = calculate_auc(sc_sim_adata, n_threads=4, n_subsamples=3, classifier=estimator, random_state=42)
    assert results["CellTypeA"][2]["subsample_idx"] == 2
    assert "augur_score" in adata.obs.columns
    assert np.allclose(results["summary_metrics"].loc["mean_augur_score"].tolist(), [0.433333, 0.666667, 0.666667])
    assert "feature_importances" in results.keys()


# Test cross validation
sc.pp.subsample(sc_sim_adata, n_obs=100, random_state=42)

rf_classifier = create_estimator("random_forest_classifier", Params(random_state=42))
lr_classifier = create_estimator("logistic_regression_classifier", Params(random_state=42))
rf_regressor = create_estimator("random_forest_regressor", Params(random_state=42))


def test_classifier():
    """Test run cross validation with classifier."""
    cv = run_cross_validation(sc_sim_adata, rf_classifier, subsample_idx=1, folds=3, random_state=42)
    auc = 0.802411
    assert any([isclose(cv["mean_auc"], auc, abs_tol=10 ** -5)])

    cv = run_cross_validation(sc_sim_adata, lr_classifier, subsample_idx=1, folds=3, random_state=42)
    auc = 0.869871
    assert any([isclose(cv["mean_auc"], auc, abs_tol=10 ** -5)])


def test_regressor():
    """Test run cross validation with regressor."""
    cv = run_cross_validation(sc_sim_adata, rf_regressor, subsample_idx=1, folds=3, random_state=42)
    ccc = 0.433445
    r2 = 0.274051
    assert any([isclose(cv["mean_ccc"], ccc, abs_tol=10 ** -5), isclose(cv["mean_r2"], r2, abs_tol=10 ** -5)])
