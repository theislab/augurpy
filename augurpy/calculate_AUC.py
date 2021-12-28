import random
from math import floor, nan
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from joblib import Parallel, delayed
from pandas import DataFrame
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from augurpy.cv import run_cross_validation


def cross_validate_subsample(
    adata: AnnData,
    estimator: Union[RandomForestRegressor, RandomForestClassifier, LogisticRegression],
    augur_mode: str,
    subsample_size: int,
    folds: int,
    feature_perc: float,
    subsample_idx: int,
    random_state: Optional[int],
) -> DataFrame:
    """Cross validate subsample anndata object.

    Args:
        adata: Anndata with obs `label` and `cell_type` for label and cell type and dummie variable `y_` columns used as target
        estimator: classifier to use in calculating augur metrics, either random forest or logistic regression
        augur_mode: one of default, velocity or permute. Setting augur_mode = "velocity" disables feature selection,
            assuming feature selection has been performed by the RNA velocity procedure to produce the input matrix,
            while setting augur_mode = "permute" will generate a null distribution of AUCs for each cell type by
            permuting the labels
        subsample_size: number of cells to subsample randomly per type from each experimental condition
        folds: number of folds to run cross validation on
        subsample_idx: index of the subsample
        random_state: set numpy random seed, sampling seed and fold seed

    Returns:
        Results for each cross validation fold.
    """
    subsample = draw_subsample(
        adata,
        augur_mode,
        subsample_size,
        feature_perc=feature_perc,
        stratified=is_classifier(estimator),
        random_state=random_state,
    )
    results = run_cross_validation(
        estimator=estimator, subsample=subsample, folds=folds, subsample_idx=subsample_idx, random_state=random_state
    )
    return results


def draw_subsample(
    adata: AnnData,
    augur_mode: str,
    subsample_size: int,
    feature_perc: float,
    stratified: bool,
    random_state: Optional[int],
) -> AnnData:
    """Subsample and select random features of anndata object.

    Args:
        adata: Anndata with obs `label` and `cell_type` for label and cell type and dummie variable `y_` columns used as target
        augur_mode: one of default, velocity or permute. Setting augur_mode = "velocity" disables feature selection,
            assuming feature selection has been performed by the RNA velocity procedure to produce the input matrix,
            while setting augur_mode = "permute" will generate a null distribution of AUCs for each cell type by
            permuting the labels
        subsample_size: number of cells to subsample randomly per type from each experimental condition
        stratified: if `True` subsamples are stratified according to condition
        random_state: set numpy random seed and sampling seed

    Returns:
        Subsample of anndata object of size subsample_size
    """
    if augur_mode == "permut":
        # shuffle labels
        y_columns = [col for col in adata.obs if col.startswith("y")]
        adata.obs[y_columns] = adata.obs[y_columns].sample(frac=1, random_state=random_state).reset_index(drop=True)

    if augur_mode == "velocity":
        # no feature selection, assuming this has already happenend in calculating velocity
        subsample = sc.pp.subsample(adata, n_obs=subsample_size, copy=True, random_state=random_state)
        return subsample

    # randomly sample features
    if random_state is not None:
        random.seed(random_state)
    features = random.sample(adata.var_names.tolist(), floor(len(adata.var_names.tolist()) * feature_perc))
    # randomly sample samples
    subsample = sc.pp.subsample(adata[:, features], n_obs=subsample_size, copy=True, random_state=random_state)
    return subsample


def average_metrics(subsample_cv_results: List[Any]) -> Dict[Any, Any]:
    """Calculate average metric of cross validation runs.

    Args:
        cross_validation_results: list of all subsample cross validations

    Returns:
        Dict containing the average result for each metric.
    """
    metric_names = [metric for metric in [*subsample_cv_results[0].keys()] if metric.startswith("mean")]
    metric_list: Dict[Any, Any] = {}
    for d in subsample_cv_results:
        for metric in metric_names:
            metric_list[metric] = metric_list.get(metric, []) + [d[metric]]

    return {metric: np.mean(values) for metric, values in metric_list.items()}


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
    random_state: Optional[int] = None,
) -> Tuple[AnnData, Dict[str, Any]]:
    """Calculates the Area under the Curve using the given classifier.

    Args:
        adata: Anndata with obs `label` and `cell_type` for label and cell type and dummie variable `y_` columns used as target
        classifier: classifier to use in calculating augur metrics, either random forest or logistic regression
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
        random_state: set numpy random seed, sampling seed and fold seed

    Returns:
        A dictionary containing the following keys: Dict[X, y, celltypes, parameters, results, feature_importances, AUC]
        and the Anndata object with additional augur_score obs and uns summary.
    """
    results: Dict[Any, Any] = {"summary_metrics": {}}
    adata.obs["augur_score"] = nan
    for cell_type in adata.obs["cell_type"].unique():
        cell_type_subsample = adata[adata.obs["cell_type"] == cell_type]
        results[cell_type] = Parallel(n_jobs=n_threads)(
            delayed(cross_validate_subsample)(
                adata=cell_type_subsample,
                estimator=classifier,
                augur_mode=augur_mode,
                subsample_size=subsample_size,
                folds=folds,
                feature_perc=feature_perc,
                subsample_idx=i,
                random_state=random_state,
            )
            for i in range(n_subsamples)
        )
        # summarize scores for cell type
        results["summary_metrics"][cell_type] = average_metrics(results[cell_type])

        # add scores as observation to anndata
        mask = adata.obs["cell_type"].str.startswith(cell_type)
        adata.obs.loc[mask, "augur_score"] = results["summary_metrics"][cell_type]["mean_augur_score"]

    results["summary_metrics"] = pd.DataFrame(results["summary_metrics"])
    adata.uns["summary_metrics"] = pd.DataFrame(results["summary_metrics"])

    return adata, results
