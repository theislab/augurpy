from math import floor
from random import sample
from typing import Any, Dict, Literal, Union

import scanpy as sc
from anndata import AnnData
from joblib import Parallel, delayed
from pandas import DataFrame
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from augurpy.cv import run_cross_validation


def cross_validate_subsample(
    input: AnnData,
    estimator: Union[RandomForestRegressor, RandomForestClassifier, LogisticRegression],
    augur_mode: str,
    subsample_size: int,
    folds: int,
    feature_perc: float,
    subsample_idx: int,
) -> DataFrame:
    """Cross validate subsample.

    Args:
        input: Pandas DataFrame
        estimator: classifier
        augur_mode: one of default, velocity or permute. Setting augur_mode = "velocity" disables feature selection,
            assuming feature selection has been performed by the RNA velocity procedure to produce the input matrix,
            while setting augur_mode = "permute" will generate a null distribution of AUCs for each cell type by
            permuting the labels
        subsample_size: number of cells to subsample randomly per type from each experimental condition
        folds: number of folds to run cross validation on
        subsample_idx: index of the subsample

    Returns:
        Results for each cross validation fold.
    """
    subsample = draw_subsample(
        input, augur_mode, subsample_size, feature_perc=feature_perc, stratified=is_classifier(estimator)
    )
    results = run_cross_validation(estimator=estimator, subsample=subsample, folds=folds, subsample_idx=subsample_idx)
    return results


def draw_subsample(
    adata: AnnData, augur_mode: str, subsample_size: int, feature_perc: float, stratified: bool
) -> AnnData:
    """Subsample input and select random features.

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
    if augur_mode == "permut":
        # shuffle labels
        y_columns = [col for col in adata.obs if col.startswith("y")]
        adata.obs[y_columns] = adata.obs[y_columns].sample(frac=1).reset_index(drop=True)

    if augur_mode == "velocity":
        # no feature selection, assuming this has already happenend in calculating velocity
        subsample = sc.pp.subsample(adata, n_obs=subsample_size, copy=True)
        return subsample

    # randomly sample features
    features = sample(adata.var_names.tolist(), floor(len(adata.var_names.tolist()) * feature_perc))
    # randomly sample samples
    subsample = sc.pp.subsample(adata[:, features], n_obs=subsample_size, copy=True)
    return subsample


def calculate_auc(
    adata: AnnData,
    classifier: Union[RandomForestClassifier, RandomForestRegressor, LogisticRegression],
    n_subsamples: int = 50,
    subsample_size: int = 20,
    folds: int = 3,
    min_cells: int = None,
    feature_perc: float = 0.5,
    n_threads: int = 4,
    show_progress: bool = True,
    augur_mode: Union[Literal["permute"], Literal["default"], Literal["velocity"]] = "default",
) -> Dict[Any, Any]:
    """Calculates the Area under the Curve using the given classifier.

    Args:
        input: Anndata with obs `label` and `cell_type` for label and cell type and dummie variable `y_` columns used as target
        n_subsamples: number of random subsamples to draw from complete dataset for each cell type
        subsample_size: number of cells to subsample randomly per type from each experimental condition
        folds: number of folds to run cross validation on
        min_cells: minimum number of cells for a particular cell type in each condition in order to retain that type for
            analysis (depricated..)
        feature_perc: proportion of genes that are randomly selected as features for input to the classifier in each
            subsample using the random gene filter
        n_threads: number of threads to use for parallelization
        show_progress: if `True` display a progress bar for the analysis with estimated time remaining
        augur_mode: one of default, velocity or permute. Setting augur_mode = "velocity" disables feature selection,
            assuming feature selection has been performed by the RNA velocity procedure to produce the input matrix,
            while setting augur_mode = "permute" will generate a null distribution of AUCs for each cell type by
            permuting the labels
        classifier: classifier to use in calculating the area under the curve either random forest or logistic regression
        params: dictionary of parameters for random forest and logistic regression

    Returns:
        A dictionary containing the following keys: Dict[X, y, celltypes, parameters, results, feature_importances, AUC]
        and the Anndata object with additional results layer.
    """
    results = {}
    for cell_type in adata.obs["cell_type"].unique():
        cell_type_subsample = adata[adata.obs["cell_type"] == cell_type]
        results[cell_type] = Parallel(n_jobs=n_threads)(
            delayed(cross_validate_subsample)(
                input=cell_type_subsample,
                estimator=classifier,
                augur_mode=augur_mode,
                subsample_size=subsample_size,
                folds=folds,
                feature_perc=feature_perc,
                subsample_idx=i,
            )
            for i in range(n_subsamples)
        )

    return results
