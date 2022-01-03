"""Calculates augur score for given dataset and estimator."""
from __future__ import annotations

import random
from collections import defaultdict
from math import floor, nan
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from joblib import Parallel, delayed
from pandas import DataFrame
from rich.progress import track
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, r2_score, roc_auc_score
from sklearn.model_selection import KFold, cross_validate


def cross_validate_subsample(
    adata: AnnData,
    estimator: RandomForestRegressor | RandomForestClassifier | LogisticRegression,
    augur_mode: str,
    subsample_size: int,
    folds: int,
    feature_perc: float,
    subsample_idx: int,
    random_state: int | None,
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
    random_state: int | None,
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


def ccc_score(y_true, y_pred) -> float:
    """Implementation of Lin's Concordance correlation coefficient, based on https://gitlab.com/-/snippets/1730605.

    Args:
        y_true: array-like of shape (n_samples), ground truth (correct) target values
        y_pred: array-like of shape (n_samples), estimated target values

    Returns:
        Concordance correlation coefficient.
    """
    # covariance between y_true and y_pred
    s_xy = np.cov([y_true, y_pred])[0, 1]
    # means
    x_m = np.mean(y_true)
    y_m = np.mean(y_pred)
    # variances
    s_x_sq = np.var(y_true)
    s_y_sq = np.var(y_pred)

    # condordance correlation coefficient
    ccc = (2.0 * s_xy) / (s_x_sq + s_y_sq + (x_m - y_m) ** 2)

    return ccc


def set_scorer(
    estimator: RandomForestRegressor | RandomForestClassifier | LogisticRegression,
) -> dict[str, Any]:
    """Set scoring fuctions for cross-validation based on estimator.

    Args:
        estimator: classifier object used to fit the model used to calculate the area under the curve

    Returns:
        Dict linking name to scorer object and string name
    """
    return (
        {"augur_score": make_scorer(roc_auc_score), "auc": make_scorer(roc_auc_score)}
        if isinstance(estimator, RandomForestClassifier) or isinstance(estimator, LogisticRegression)
        else {"augur_score": make_scorer(ccc_score), "r2": make_scorer(r2_score), "ccc": make_scorer(ccc_score)}
    )


def run_cross_validation(
    subsample: AnnData,
    estimator: RandomForestRegressor | RandomForestClassifier | LogisticRegression,
    subsample_idx: int,
    folds: int = 3,
    random_state=Optional[int],
) -> dict:
    """Perform cross validation on given subsample.

    Args:
        subsample: subsample of gene expression matrix of size subsample_size
        estimator: classifier object to use in calculating the area under the curve
        subsample_idx: index of subsample
        folds: number of folds
        random_state: set random fold seed

    Returns:
        Dictionary containing prediction metrics and estimator for each fold.
    """
    scorer = set_scorer(estimator)
    x = subsample.to_df()
    y = subsample.obs[[col for col in subsample.obs if col.startswith("y")]]
    folds = KFold(n_splits=folds, random_state=random_state, shuffle=True)

    results = cross_validate(
        estimator=estimator,
        X=x,
        y=y.values.ravel(),
        scoring=scorer,
        cv=folds,
        return_estimator=True,
    )

    results["subsample_idx"] = subsample_idx
    for score in scorer.keys():
        results[f"mean_{score}"] = results[f"test_{score}"].mean()

    # feature importances
    feature_importances = defaultdict(list)
    if isinstance(estimator, RandomForestClassifier) or isinstance(estimator, RandomForestRegressor):
        for fold, estimator in list(zip(range(len(results["estimator"])), results["estimator"])):
            feature_importances["genes"].extend(x.columns.tolist())
            feature_importances["feature_importances"].extend(estimator.feature_importances_.tolist())
            feature_importances["subsample_idx"].extend(len(x.columns) * [subsample_idx])
            feature_importances["fold"].extend(len(x.columns) * [fold])

    if isinstance(estimator, LogisticRegression):
        pass

    results["feature_importances"] = feature_importances

    return results


def average_metrics(cell_cv_results: list[Any]) -> dict[Any, Any]:
    """Calculate average metric of cross validation runs done of one cell type.

    Args:
        cell_cv_results: list of all cross validation runs of one cell type

    Returns:
        Dict containing the average result for each metric of one cell type
    """
    metric_names = [metric for metric in [*cell_cv_results[0].keys()] if metric.startswith("mean")]
    metric_list: dict[Any, Any] = {}
    for subsample_cv_result in cell_cv_results:
        for metric in metric_names:
            metric_list[metric] = metric_list.get(metric, []) + [subsample_cv_result[metric]]

    return {metric: np.mean(values) for metric, values in metric_list.items()}


def calculate_auc(
    adata: AnnData,
    classifier: RandomForestClassifier | RandomForestRegressor | LogisticRegression,
    n_subsamples: int = 50,
    subsample_size: int = 20,
    folds: int = 3,
    min_cells: int = None,
    feature_perc: float = 0.5,
    n_threads: int = 4,
    show_progress: bool = True,
    augur_mode: Literal["permute"] | Literal["default"] | Literal["velocity"] = "default",
    random_state: int | None = None,
) -> tuple[AnnData, dict[str, Any]]:
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
    results: dict[Any, Any] = {"summary_metrics": {}, "feature_importances": defaultdict(list)}
    adata.obs["augur_score"] = nan
    for cell_type in track(adata.obs["cell_type"].unique(), description="Processing data."):
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

        # concatenate feature importances for each subsample cv
        subsample_feature_importances_dicts = [cv["feature_importances"] for cv in results[cell_type]]

        for dictionary in subsample_feature_importances_dicts:
            for key, value in dictionary.items():
                results["feature_importances"][key].extend(value)
        results["feature_importances"]["CellType"].extend(
            [cell_type]
            * (len(results["feature_importances"]["genes"]) - len(results["feature_importances"]["CellType"]))
        )

    results["summary_metrics"] = pd.DataFrame(results["summary_metrics"])
    results["feature_importances"] = pd.DataFrame(results["feature_importances"])
    adata.uns["summary_metrics"] = pd.DataFrame(results["summary_metrics"])

    return adata, results
