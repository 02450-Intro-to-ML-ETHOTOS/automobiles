from classification_transform_data import * # imports numpy as np, pandas as pd, various sklearn modules
from classification_baseline_model import *
from classification_logreg_model import *
from classification_tree_model import *
from mcnemar_test import mcnemar


from pprint import pprint
import scipy.stats as st
from itertools import combinations
import json

# set up data structures for storing errors and parameters using dict comprehension
models = ["B", "RLR", "CT"] # baseline, Regularized Logistic Regresison, Classification Tree
# this will make a dict with a list for each model to contain the test error for each fold
model_errors_test = {m: [] for m in models}
model_parameters = {m: [] for m in models}
model_predictions = {m: [] for m in models}

y = onehot2classidx(y)

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    error_rate = np.sum(y_pred!=y) / len(y)
    return error_rate

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
    N_test, M_test = X_test.shape

    # general workflow for each model
    # 1) fit
    # 2) predict
    # 3) calculate error (Ei_test)
    # 4) save error and parameter

    # baseline model
    base_model = ClassificationBaselineModel()
    base_model.fit(y_train)
    model_parameters["B"].append(base_model.y_pred)
    model_errors_test["B"].append(evaluate_model(base_model, X_test, y_test))
    model_predictions["B"].append(base_model.predict(X_test))

    # regularized logistic regression model
    lambdas = np.logspace(-2, 2, 32)
    rlr_model = ClassificationLogisticRegressionModel()
    rlr_model.fit(X_train, y_train, lambdas, 10)
    model_parameters["RLR"].append(rlr_model.lambda_opt)
    model_errors_test["RLR"].append(evaluate_model(rlr_model, X_test, y_test))
    model_predictions["RLR"].append(rlr_model.predict(X_test))

    # classification tree model
    criteria = np.arange(2, 20, 1)
    ct_model = ClassificationTreeModel()    
    ct_model.fit(X_train, y_train, criteria, 10)
    model_parameters["CT"].append(ct_model.criteria_opt)
    model_errors_test["CT"].append(evaluate_model(ct_model, X_test, y_test))
    model_predictions["CT"].append(ct_model.predict(X_test))

# print(f"Model errors across {K} outer folds:")
# pprint(model_errors_test)

# print(f"Model parameters across {K} outer folds:")
# pprint(model_parameters)

# print(f"Model predictions across {K} outer folds:")
# pprint(model_predictions)


# statistical comparison - setup I - McNemar's test
alpha = 0.05
model_combinations = list(combinations(models, 2))
model_combinations = model_combinations + [(b,a) for a,b in model_combinations]
tests = {}

for mA, mB in model_combinations:
    print(f"Comparing: {mA} and {mB}")
    # 1) extract predictions
    yA = np.concatenate(model_predictions[mA])
    yB = np.concatenate(model_predictions[mB])

    # 2) feed to test func
    p, ci, theta, nn = mcnemar(y, yA, yB, alpha)
    print(f"p = {p}, theta = {theta}, with CI = {ci}")
    print("Comparison matrix n\n", nn)

    # 3) print conclusion
    if p < 0.05:
        print(f"H1: models {mA} and {mB} have different performance, Z != 0")
    else: # p >= 0.05
        print(f"H0: models {mA} and {mB} have the same performance, Z = 0")

    if theta > 0:
        print(f"Positive theta: {mA} is preferable over {mB}")
    else:
        print(f"Negative theta: {mB} is preferable over {mA}")

    # 4) save results
    tests[f"{mA}-vs-{mB}"] = {"p": p, "theta": theta, "CI": ci, "nn": nn.tolist()}


# dump output of comparison
with open('../out/model_comparisons/classification.json', 'w') as fp:
    data = {"errors_test": model_errors_test, 
            "parameters": {k: [float(e) for e in l] for k,l in model_parameters.items()},
            "statistical_tests": tests}
    json.dump(data, fp, sort_keys=True, indent=4)

