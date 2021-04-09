from regression_transform_data import *
from regression_baseline_model import *
from regression_regularized_model import *
# imports numpy as np, pandas as pd, various sklearn modules

from pprint import pprint

# set up cross validation for outer folds
K = 10
# we set random_state to ensure reproducibility
CV = model_selection.KFold(K, shuffle=True, random_state=42)

error_train = np.empty((K,1))

# set up data structures for storing errors and parameters using dict comprehension
models = ["B", "ANN", "RR"] # baseline, artificial neural network, ridge regression
model_errors_test = {k: [] for k in models} # this will make a dict with a list for each model
model_parameters = {k: [] for k in models}

error_test = np.empty((K,1))

for k, (train_index, test_index) in enumerate(CV.split(X, y)): # use enumerate to get index k
    # Let Dk^train, Dk^test be the k'th split of D
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    N_test, M_test = X_test.shape

    # general workflow for each model
    # 1) fit
    # 2) predict
    # 3) calculate error (Ei_test)
    # 4) save error and parameter

    # baseline model
    basic_model = RegressionBaselineModel()
    basic_model.fit(y_train)
    y_pred_basic_model = basic_model.predict(X_test)
    err_sq_basic_model = np.power((y_test - y_pred_basic_model), 2).sum() / N_test
    model_errors_test["B"].append(err_sq_basic_model)

    # ridge regression model
    lambdas = np.logspace(-2, 2, 32)
    rr_model = RidgeRegressionModel()
    rr_model.fit(X_train, y_train, lambdas, 10)
    y_pred_rr_model = rr_model.predict(X_test)
    err_sq_rr_model = np.power((y_test - y_pred_rr_model), 2).sum() / N_test
    model_errors_test["RR"].append(err_sq_rr_model)
    model_parameters["RR"].append(rr_model.lambda_opt)
    
pprint(model_errors_test)
pprint(model_parameters)
