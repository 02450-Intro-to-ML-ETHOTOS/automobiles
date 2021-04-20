from regression_transform_data import * # imports numpy as np, pandas as pd, various sklearn modules
from regression_baseline_model import *
from regression_regularized_model import *
from regression_ann_model import *
from t_test_paired import t_test_paired

import torch

from pprint import pprint
import scipy.stats as st
from itertools import combinations
import json


# set up data structures for storing errors and parameters using dict comprehension
models = ["B", "ANN", "RR"] # baseline, artificial neural network, ridge regression
# this will make a dict with a list for each model to contain the test error for each fold
model_errors_test = {m: [] for m in models}
model_parameters = {m: [] for m in models}
model_predictions = {m: [] for m in models}
ys = []

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    error_squared = np.power((y - y_pred), 2).sum() / X.shape[0]
    return error_squared

# set up cross validation for outer folds
K = 10
# we set random_state to ensure reproducibility
CV = model_selection.KFold(K, shuffle=True, random_state=42)
for k, (train_index, test_index) in enumerate(CV.split(X, y)): # use enumerate to get index k
    print(f"Outer CV fold: {k+1}/{K}")
    # TODO: should we normalize internally? Mention in report
    # Let Dk^train, Dk^test be the k'th split of D
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    ys.append(y_test)
    N_test, M_test = X_test.shape

    # general workflow for each model
    # 1) fit
    # 2) predict
    # 3) calculate error (Ei_test)
    # 4) save error and parameter

    # baseline model
    base_model = RegressionBaselineModel()
    base_model.fit(y_train)
    model_parameters["B"].append(base_model.y_pred)
    model_errors_test["B"].append(evaluate_model(base_model, X_test, y_test))
    model_predictions["B"].append(base_model.predict(X_test))

    # ridge regression model
    lambdas = np.logspace(-2, 1, 32) # np.logspace(-2, 2, 4) # np.logspace(-2, 2, 32)
    rr_model = RidgeRegressionModel()
    rr_model.fit(X_train, y_train, lambdas, 10)
    model_parameters["RR"].append(rr_model.lambda_opt)
    model_errors_test["RR"].append(evaluate_model(rr_model, X_test, y_test))
    model_predictions["RR"].append(rr_model.predict(X_test))
    
    # ann model
    n_hidden = [1, 16, 128] # , 256, 512, 64, 128, 256, 512, 4096
    ann_model = RegressionANNModel()
    ann_model.fit(torch.Tensor(X_train), torch.Tensor(y_train), n_hidden, 10, max_iter=4000)
    model_errors_test["ANN"].append(evaluate_model(ann_model, torch.Tensor(X_test), y_test))
    model_parameters["ANN"].append(ann_model.n_hidden)
    model_predictions["ANN"].append(ann_model.predict(torch.Tensor(X_test)))


# print(f"Model errors across {K} outer folds:")
# pprint(model_errors_test)

# print(f"Model parameters across {K} outer folds:")
# pprint(model_parameters)

# statistical comparison - setup I - paired t-test
# N.B. Include p-values and conﬁdence intervals for the three pairwise tests
model_combinations = list(combinations(models, 2))
tests = {}

for mA, mB in model_combinations:
    print(f"Comparing: {mA} and {mB}")
    # 1) extract errors
    zA = np.concatenate(model_predictions[mA])
    zB = np.concatenate(model_predictions[mB])

    # 2) feed to t_test func
    p, ci = t_test_paired(zA, zB)
    print(f"p = {p}, with CI: {ci}")

    # 3) print conclusion
    if p < 0.05:
        print(f"H1: models {mA} and {mB} have different performance, Z != 0")
    else: # p >= 0.05
        print(f"H0: models {mA} and {mB} have the same performance, Z = 0")

    # 4) save results
    tests[f"{mA}-vs-{mB}"] = {"p": p, "CI": ci}


# dump output of comparison
with open('../out/model_comparisons/regression.json', 'w') as fp:
    data = {"errors_test": model_errors_test, 
            "parameters": {k: [float(e) for e in l] for k,l in model_parameters.items()},
            "statistical_tests": tests}
    json.dump(data, fp, sort_keys=True, indent=4)


import pandas as pd
# save error rates
df = pd.DataFrame({"B": model_errors_test["B"], "RR": model_errors_test["RR"], "ANN": model_errors_test["ANN"]})
df.to_csv("../out/model_comparisons/regression_errors.csv")

# save predictions
df = pd.DataFrame({"Y": np.concatenate(ys), "B":np.concatenate(model_predictions["B"]), "RR":np.concatenate(model_predictions["RR"]), "ANN":np.concatenate(model_predictions["ANN"])})
df.to_csv("../out/model_comparisons/regression_predictions.csv")
