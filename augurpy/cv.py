def cross_validate(subsample: DataFrame, estimator: any[RandomForest, LogisticRegression], metrics: list, folds: int, subsample_idx: int) -> list(subsample_idx, results, feature_importances):
    """Perform cross validation on given subsample
    Args: 
        subsample: subsample of gene expression matrix of size subsample_size
        estimator: classifier object to use in calculating the area under the curve
        metrics: list of metrics to measure in each cv-fold
        folds: number of folds
        subsample_idx: index of subsample

    Returns: 
        Data frame containing prediction metrics for each fold 

    """


