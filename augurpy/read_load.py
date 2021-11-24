from typing import Union

from anndata import AnnData
from pandas import DataFrame


def load(input: Union[AnnData, DataFrame], meta: DataFrame, label_col: str, cell_type_col: str) -> DataFrame:
    """Loads the input data.

    Args:
        input: Anndata or matrix containing gene expression values (genes in rows, cells in columns) and optionally meta
            data about each cell.
        meta: Optional data frame containing meta data about each cell.
        label_col: column of the meta data frame or the anndata or matrix containing the condition labels for each cell
            in the gene-by-cell expression matrix
        cell_type_col: column of the meta data frame or the anndata or matrix containing the cell type labels for each
            cell in the gene-by-cell expression matrix


    Returns:
        DataFrame containing gene expression values (genes in rows, cells in columns), cell type and condition

    """


def feature_selection(input: DataFrame, var_quantile: float) -> DataFrame:
    """Feature selection by variance.

    Args:
        input: DataFrame containing gene expression values (genes in rows, cells in columns)
         var_quantile: quantile of highly variable genes to retain for each cell type using the variable gene filter

    Results:
        DataFrame with selected features
    """
