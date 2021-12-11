from typing import Optional, Union

from anndata import AnnData
from pandas import DataFrame
from scanpy.preprocessing import highly_variable_genes
from rich import print


def load(
    input: Union[AnnData, DataFrame],
    meta: Optional[DataFrame] = None,
    label_col: str = "label_col",
    cell_type_col: str = "cell_type_col",
) -> AnnData:
    """Loads the input data.

    Args:
        input: Anndata or matrix containing gene expression values (genes in rows, cells in columns) and optionally meta
            data about each cell.
        meta: Optional Pandas DataFrame containing meta data about each cell.
        label_col: column of the meta DataFrame or the Anndata or matrix containing the condition labels for each cell
            in the cell-by-gene expression matrix
        cell_type_col: column of the meta DataFrame or the Anndata or matrix containing the cell type labels for each
            cell in the cell-by-gene expression matrix

    Returns:
        Anndata object containing gene expression values (cells in rows, genes in columns) and cell type, label as obs
    """
    if isinstance(input, AnnData):
        input.obs = input.obs.rename(columns={cell_type_col: "cell_type", label_col: "label"})
        adata = input

    elif isinstance(input, DataFrame):
        if meta is None:
            try:
                cell_type = input[cell_type_col]
                label = input[label_col]
            except KeyError:
                print("[bold red]No column names matching cell_type_col and label_col.")

        label = input[label_col] if meta is None else meta[label_col]
        cell_type = input[cell_type_col] if meta is None else meta[cell_type_col]

        adata = AnnData(X=input, obs={"cell_type": cell_type, "label": label})

    adata = feature_selection(adata)

    return adata


def feature_selection(input: AnnData) -> AnnData:
    """Feature selection by variance.

    Args:
        input: Pandas DataFrame containing gene expression values (cells in rows, genes in columns)

    Results:
        Anndata object with highly variable genes added as layer
    """
    min_features_for_selection = 1000

    if len(input.var_names) - 2 > min_features_for_selection:
        highly_variable_genes(input)

    return input
