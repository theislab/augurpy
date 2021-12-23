import scanpy as sc

from augurpy.calculate_AUC import calculate_auc
from augurpy.estimators import Params, create_estimator
from augurpy.read_load import load

sc_sim_adata = sc.read_h5ad("tests/sc_sim.h5ad")
estimator = create_estimator("random_forest_classifier", Params(random_state=42))


def test_calculate_auc():
    """Tests auc calculation."""
    adata, results = calculate_auc(adata=load(sc_sim_adata), n_threads=4, n_subsamples=3, classifier=estimator)
    assert results["CellTypeA"][2]["subsample_idx"] == 2
    assert "augur_score" in adata.obs.columns
