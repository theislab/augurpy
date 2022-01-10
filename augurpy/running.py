# %%
from math import isclose
from pathlib import Path

import numpy as np
import scanpy as sc

from augurpy.estimator import Params, create_estimator
from augurpy.evaluate import predict, run_cross_validation, draw_subsample
from augurpy.read_load import load

sc_sim_adata = sc.read_h5ad("../tests/sc_sim.h5ad")
sc_sim_adata = load(sc_sim_adata)

# estimators
rf_classifier = create_estimator("random_forest_classifier")
lr_classifier = create_estimator("logistic_regression_classifier")
rf_regressor = create_estimator("random_forest_regressor",)

adata = draw_subsample(adata=sc_sim_adata, augur_mode='default', subsample_size=20, feature_perc=0.5, categorical=True, random_state=42)

cv = run_cross_validation(adata, rf_classifier, subsample_idx=1, random_state=None, folds=3)

print(cv['mean_augur_score'])

permut_subsample = draw_subsample(adata=sc_sim_adata, augur_mode='permut', subsample_size=20, feature_perc=0.5, categorical= True, random_state=42)


# %%
from math import isclose
from pathlib import Path

import numpy as np
import scanpy as sc

from augurpy.estimator import Params, create_estimator
from augurpy.evaluate import predict, run_cross_validation, draw_subsample
from augurpy.read_load import load

sc_sim_adata = sc.read_h5ad("../tests/sc_sim.h5ad")
sc_sim_adata = load(sc_sim_adata)

# estimators
rf_classifier = create_estimator("random_forest_classifier")
lr_classifier = create_estimator("logistic_regression_classifier")
rf_regressor = create_estimator("random_forest_regressor")

adata, results = predict(sc_sim_adata, n_threads=4, n_subsamples=3, classifier=rf_classifier, random_state=None)
#ad, res = predict(sc_sim_adata, n_threads=4, classifier=rf_regressor, random_state=None)
print(results['summary_metrics'])




# %%
import numpy as np
for r in [results['CellTypeA'], results['CellTypeB'], results['CellTypeC']]:
    for d in r:
        #print(d['mean_augur_score'])
        if np.all(d['test_precision']): 
            
            print(dict(list(d.items())[3:14]))
            break



# %%
from math import isclose
from pathlib import Path

import numpy as np
import scanpy as sc

from augurpy.estimator import Params, create_estimator
from augurpy.evaluate import predict, run_cross_validation, draw_subsample
from augurpy.read_load import load

