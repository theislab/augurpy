# This will be one of the main functions called by the user
# It takes all the arguments like in the R version. 
#
# - check number of folds/subsample size are compatible (n > 1 in every fold)
# - check augur mode -> run different function depending on mode


# data = read_load()
# cv = cv()
# scorer = scorer()
# estimator = estimator()

# calculate_AUC for each subsample using cross validation
# (hier bin ich mir nicht sicher wo das hinmuss, muss auch nochmal nachlesen wie viel sklearn schon macht..)



def calculate_AUC(input: any[AnnData, NumpyArray], meta: MetaData, label_col: str, cell_type_col: str, 
    n_subsamples: int, subsample_size: int, folds: int, min_cells: int, var_quantile: float, feature_perc: float, 
    n_threads: int, show_progress: bool, augur_mode: any['default', 'velocity', 'permute'], classifier: any[RandomForest, LogisticRegression], rf_params: list, lr_params: list) -> list[X, y, celltypes, parameters, results, feature_importances, AUC]:
    
    """Calculates the Area under the Curve of a trained random forest.
    Args:
        input: Anndata or matrix containing gene expression values (genes in rows, cells in columns) and optionally meta data about each cell.
        meta: Optional data frame containing meta data about each cell.
        label_col: column of the meta data frame or the anndata or matrix containing the condition labels for each cell in the gene-by-cell expression matrix
        cell_type_col: column of the meta data frame or the anndata or matrix containing the cell type labels for each cell in the gene-by-cell expression matrix
        n_subsamples: number of random subsampels to draw from complete dataset for each cell type
        subsample_size: number of cells to subsample randomly per type from each experimental condition
        folds: number of folds of cross-validation to run
        min_cells: minumun number of cells for a particula cell type in each condition in order to retain that type for analysis (depricated..)
        var_quantile: quantile of highly variable genes to retain for each cell type using the variable gene filter
        feature_perc: proportion of genes that are randomly selected as features for input to the classifier in each subsample using the random gene filter
        n_threads: number of threads to use for parallelization
        show_progress: if TRUE display a progress bar for the analysis with estimated time remaining
        augur_mode: one of default, velocity or permute. Setting augur_mode = "velocity" disables feature selection, assuming feature selection has been performed by the RNA 
            velocity procedure to produce the input matrix, while setting augur_mode = "permute" will generate a null distribution of AUCs for each cell type by permuting the labels
        classifier: classifier to use in claculating the area under the curve either random forest or logistic regression
        rf_params: list of parameters for random forest
        lr_params: list of parameters for logistic regression

    Returns:
        The area under the curve of a random forest evaluated by X
    """



def subsample(input: DataFrame, augur_mode: any['default', 'velocity', 'permute'], subsample_size: int, stratified: bool) -> DataFrame:
    """Subsample 
    Args: 
        input: DataFrame containing gene expression values (genes in rows, cells in columns) along with cell type and condition
        augur_mode: one of default, velocity or permute. Setting augur_mode = "velocity" disables feature selection, assuming feature selection has been performed by the RNA 
            velocity procedure to produce the input matrix, while setting augur_mode = "permute" will generate a null distribution of AUCs for each cell type by permuting the labels
        subsample_size: number of cells to subsample randomly per type from each experimental condition
        stratified: if TRUE subsamples are stratified according to condition
    
    Returns:
        Subsample of input of size subsample_size
    """ 

