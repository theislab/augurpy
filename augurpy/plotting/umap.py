import anndata
import scanpy as sc


def plot_umap(adata: anndata.AnnData):
    """Plot UMAP representation of anndata with augur_score labeling."""
    try:
        sc.pl.umap(adata=adata, color="augur_score")
    except KeyError:
        print(
            "[Bold yellow]Missing UMAP in obsm. Calculating UMAP using default pp.neighbors() and tl.umap() from scanpy."
        )
        sc.pp.neighbors(adata, use_rep="X")
        sc.tl.umap(adata)

    return adata
