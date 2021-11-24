from typing import Union

from anndata import AnnData
from pandas import DataFrame


def load(input: Union[AnnData, DataFrame], meta: DataFrame, label_col: str, cell_type_col: str) -> DataFrame:
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
    pass


def feature_selection(input: DataFrame, var_quantile: float) -> DataFrame:
    """Feature selection by variance.

    Args:
        input: Pandas DataFrame containing gene expression values (cells in rows, genes in columns)
         var_quantile: quantile of highly variable genes to retain for each cell type using the variable gene filter

    Results:
        Pandas DataFrame with selected features
    """
    pass
