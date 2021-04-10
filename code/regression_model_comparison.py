from regression_transform_data import *
from regression_baseline_model import *
from regression_regularized_model import *
# imports numpy as np, pandas as pd, various sklearn modules

from pprint import pprint
import scipy.stats as st
from itertools import combinations

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

# statistical comparison - setup I - paired t-test
# N.B. Include p-values and conﬁdence intervals for the three pairwise tests

def t_test_paired(zA, zB, alpha = 0.05):
    """Paired T-test

    Assumes equal sample sizes and variance

    Paired, two-tailed t-test, i.e. the hypotheses are:
    * H0: model A and B have the same perforamnce, Z = 0
    * H1: model A and B have different performance, Z != 0

    As such, the p-value indicates that the means are different in either "direction".
    This means that the comparison B vs. RR is equivalent to RR vs. B.

    Verbose implementation, clarity > fancy one-liners.
    See ex7_2_1.py for inspiration.

    TODO:
    * report - consider these aspects
    * is zA supposed to be the mean E_test error? Or the error for each observation?
    In the book box 11.3.4 they write that they recommend n >= 30 samples
    """

    z = zA - zB
    n = len(z)
    z_hat = np.sum(z) / n
    # sigma^2 is the variance
    sigma_tilde_sq = np.sum((z - z_hat)**2) / n*(n-1) # TODO: why this form of the variance?
    # sqrt(sigma^2) is the std.dev., needed for computing the Std.err. of the mean, SEM
    sigma_hat = np.sqrt(sigma_tilde_sq) # std.dev.
    sem = sigma_hat / np.sqrt(n)
    
    # p for two-tailed T-test is: P(T > |t|) = 2 * (1 - cdf(t))
    p = 2 * (1 - st.t.cdf(np.abs( z_hat / sem ), df=n-1)) # TODO: ask TA about the df?
    
    # in scipy.stats, loc is the mean and scale is the std.dev.
    CI = st.t.interval(1-alpha, df=n-1, loc=z_hat, scale=sigma_hat)  # Confidence interval

    return p, CI

# p, ci = t_test_paired(np.random.normal(0, 1, 10), np.random.normal(10, 1, 10)) # test
p, ci = t_test_paired(np.array(model_errors_test["B"]), np.array(model_errors_test["RR"]))
print(f"p = {p}, with CI: {ci}")

model_combinations = list(combinations(models, 2))

for mA, mB in model_combinations:
    print(mA, mB)
    # 1) extract errors

    # 2) feed to t_test func
