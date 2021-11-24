from typing import Dict, List, Optional, Tuple, Union

from anndata import AnnData
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression


def calculate_auc(
    input: Union[AnnData, DataFrame],
    meta: Optional[DataFrame] = None,
    label_col: str = "label",
    cell_type_col: str = "cell_type",
    n_subsamples: int = 50,
    subsample_size: int = 20,
    folds: int = 3,
    min_cells: int = None,
    var_quantile: float = 0.5,
    feature_perc: float = 0.5,
    n_threads: int = 4,
    show_progress: bool = True,
    augur_mode: str = "default",
    classifier: str = "rf",
    rf_params: Dict = None,
    lr_params: List = None,
) -> Tuple[Union[Dict, AnnData]]:
    """Calculates the Area under the Curve using the given classifier.

    Args:
        input: Anndata or matrix containing gene expression values (cells in rows, genes in columns) and optionally meta
            data about each cell.
        meta: Optional Pandas DataFrame containing meta data about each cell.
        label_col: column of the meta data or the Anndata or matrix containing the condition labels for each cell
            in the cell-by-gene expression matrix
        cell_type_col: column of the meta DataFrame or the Anndata or matrix containing the cell type labels for each
            cell in the cell-by-gene expression matrix
        n_subsamples: number of random subsamples to draw from complete dataset for each cell type
        subsample_size: number of cells to subsample randomly per type from each experimental condition
        folds: number of folds to run cross validation on
        min_cells: minimum number of cells for a particular cell type in each condition in order to retain that type for
            analysis (depricated..)
        var_quantile: quantile of highly variable genes to retain for each cell type using the variable gene filter
        feature_perc: proportion of genes that are randomly selected as features for input to the classifier in each
            subsample using the random gene filter
        n_threads: number of threads to use for parallelization
        show_progress: if `True` display a progress bar for the analysis with estimated time remaining
        augur_mode: one of default, velocity or permute. Setting augur_mode = "velocity" disables feature selection,
            assuming feature selection has been performed by the RNA velocity procedure to produce the input matrix,
            while setting augur_mode = "permute" will generate a null distribution of AUCs for each cell type by
            permuting the labels
        classifier: classifier to use in calculating the area under the curve either random forest or logistic regression
        rf_params: list of parameters for random forest
        lr_params: list of parameters for logistic regression

    Returns:
        A dictionary containing the following keys: Dict[X, y, celltypes, parameters, results, feature_importances, AUC]
        and the Anndata object with additional results layer.
    """
    pass


def draw_subsample(input: DataFrame, augur_mode: str, subsample_size: int, stratified: bool) -> DataFrame:
    """Subsample input.

    Args:
        input: Pandas DataFrame containing gene expression values (cells in rows, genes in columns) along with cell type and
            condition
        augur_mode: one of default, velocity or permute. Setting augur_mode = "velocity" disables feature selection,
            assuming feature selection has been performed by the RNA velocity procedure to produce the input matrix,
            while setting augur_mode = "permute" will generate a null distribution of AUCs for each cell type by
            permuting the labels
        subsample_size: number of cells to subsample randomly per type from each experimental condition
        stratified: if `True` subsamples are stratified according to condition

    Returns:
        Subsample of input of size subsample_size
    """
    pass


def subsample_cross_validate(
    input: DataFrame,
    augur_mode: str,
    subsample_size: int,
    stratified: bool,
    estimator: Union[RandomForestRegressor, RandomForestClassifier, LogisticRegression],
    metrics: list,
    folds: int,
    subsample_idx: int,
) -> DataFrame:
    """Cross validate subsample.

    Args:
        input: Pandas DataFrame
        augur_mode: one of default, velocity or permute. Setting augur_mode = "velocity" disables feature selection,
            assuming feature selection has been performed by the RNA velocity procedure to produce the input matrix,
            while setting augur_mode = "permute" will generate a null distribution of AUCs for each cell type by
            permuting the labels
        subsample_size: number of cells to subsample randomly per type from each experimental condition
        stratified: if the sampling for cross validation is stratified or not
        estimator: classifier
        metrics: metrics to evaluate the cross validation
        folds: number of folds to run cross validation on
        subsample_idx: index of the subsample

    Returns:
        Results for each cross validation fold.
    """
    pass
