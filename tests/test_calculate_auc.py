import scanpy as sc

from augurpy.calculate_AUC import calculate_auc
from augurpy.estimators import create_estimator
from augurpy.read_load import load

adata = sc.read_h5ad("tests/sc_sim.h5ad")
estimator = create_estimator("random_forest_classifier")


def test_calculate_auc():
    """Tests auc calculation."""
    results = calculate_auc(adata=load(adata), n_threads=4, n_subsamples=3, classifier=estimator)
    assert results["CellTypeA"][2]["subsample_idx"] == 2
