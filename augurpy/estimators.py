def generate_estimator(classifier: any[RandomForest, LogisticRegression], rf_params: list, lr_params: list) -> any[RandomForest, LogisticRegression]:
    """Estimator object
    Args: 
        classifier: classifier to use in claculating the area under the curve either random forest or logistic regression
        rf_params: list of parameters for random forest
        lr_params: list of parameters for logistic regression
    
    Returns: 
        Estimator object that classifies.

    """