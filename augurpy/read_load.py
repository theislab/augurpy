from typing import Optional, Union

import anndata
from anndata import AnnData
from pandas import DataFrame

# from scanpy.preprocessing import highly_variable_genes


def load(
    input: Union[AnnData, DataFrame],
    meta: Optional[DataFrame] = None,
    label_col: str = "label_col",
    cell_type_col: str = "cell_type_col",
    var_quantile: float = 0.5,
) -> DataFrame:
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
        Pandas DataFrame containing gene expression values (cells in rows, genes in columns), cell type and condition
    """
    if isinstance(input, anndata.AnnData):
        out = input.to_df()
        out["cell_type"] = input.obs[cell_type_col]
        out["label"] = input.obs[label_col]
        print("This is anndata.")

    if isinstance(input, DataFrame):
        # check if celltype and label columns are there, check meta data.
        out = input.rename(columns={cell_type_col: "cell_type", label_col: "label"})

    out = feature_selection(out, var_quantile)

    return out


def feature_selection(input: DataFrame, var_quantile: float) -> DataFrame:
    """Feature selection by variance.

    Args:
        input: Pandas DataFrame containing gene expression values (cells in rows, genes in columns)
         var_quantile: quantile of highly variable genes to retain for each cell type using the variable gene filter

    Results:
        Pandas DataFrame with selected features
    """
    min_features_for_selection = 1000
    if len(input.columns) - 2 > min_features_for_selection:
        # selected_genes = highly_variable_genes(input)
        pass

    return input
