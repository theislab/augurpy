import scanpy as sc

from augurpy.read_load import load

adata = sc.read_h5ad("tests/sc_sim.h5ad")


def test_load():
    """Test if load function creates anndata objects."""
    loaded_adata = load(adata)
    loaded_df = load(adata.to_df(), meta=adata.obs, cell_type_col="cell_type", label_col="label")

    assert loaded_adata.obs["y_treatment"].equals(loaded_df.obs["y_treatment"]) is True
    assert adata.to_df().equals(loaded_adata.to_df()) is True and adata.to_df().equals(loaded_df.to_df())
