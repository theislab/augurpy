from math import isclose
from pathlib import Path

import numpy as np
import scanpy as sc

from augurpy.estimator import Params, create_estimator
from augurpy.evaluate import calculate_auc, run_cross_validation
from augurpy.read_load import load

CWD = Path(__file__).parent.resolve()

sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
sc_sim_adata = load(sc_sim_adata)

# estimators
rf_classifier = create_estimator("random_forest_classifier", Params(random_state=42))
lr_classifier = create_estimator("logistic_regression_classifier", Params(random_state=42))
rf_regressor = create_estimator("random_forest_regressor", Params(random_state=42))


def test_random_forest_classifier():
    """Tests random forest for auc calculation."""
    adata, results = calculate_auc(sc_sim_adata, n_threads=4, n_subsamples=3, classifier=rf_classifier, random_state=42)
    assert results["CellTypeA"][2]["subsample_idx"] == 2
    assert "augur_score" in adata.obs.columns
    assert np.allclose(results["summary_metrics"].loc["mean_augur_score"].tolist(), [0.433333, 0.666667, 0.666667])
    assert "feature_importances" in results.keys()


def test_logistic_regression_classifier():
    """Tests logistic classifier for auc calculation."""
    adata, results = calculate_auc(sc_sim_adata, n_threads=4, n_subsamples=3, classifier=lr_classifier, random_state=42)
    assert "augur_score" in adata.obs.columns
    assert np.allclose(results["summary_metrics"].loc["mean_augur_score"].tolist(), [0.622222, 0.916667, 1.0])
    assert "feature_importances" in results.keys()


def test_random_forest_regressor():
    """Tests random forest regressor for ccc calculation."""
    adata, results = calculate_auc(sc_sim_adata, n_threads=4, n_subsamples=3, classifier=rf_regressor, random_state=42)
    assert "augur_score" in adata.obs.columns
    assert np.allclose(results["summary_metrics"].loc["mean_augur_score"].tolist(), [0.192365, 0.485231, 0.757517])
    assert np.allclose(results["summary_metrics"].loc["mean_r2"].tolist(), [-0.416232, 0.170226, 0.540434])
    assert "feature_importances" in results.keys()


# Test cross validation
def test_classifier(adata=sc_sim_adata):
    """Test run cross validation with classifier."""
    adata = sc.pp.subsample(adata, n_obs=100, random_state=42, copy=True)

    cv = run_cross_validation(adata, rf_classifier, subsample_idx=1, folds=3, random_state=42)
    auc = 0.802411
    assert any([isclose(cv["mean_auc"], auc, abs_tol=10 ** -5)])

    cv = run_cross_validation(adata, lr_classifier, subsample_idx=1, folds=3, random_state=42)
    auc = 0.869871
    assert any([isclose(cv["mean_auc"], auc, abs_tol=10 ** -5)])


def test_regressor(adata=sc_sim_adata):
    """Test run cross validation with regressor."""
    adata = sc.pp.subsample(adata, n_obs=100, random_state=42, copy=True)

    cv = run_cross_validation(adata, rf_regressor, subsample_idx=1, folds=3, random_state=42)
    ccc = 0.433445
    r2 = 0.274051
    assert any([isclose(cv["mean_ccc"], ccc, abs_tol=10 ** -5), isclose(cv["mean_r2"], r2, abs_tol=10 ** -5)])
