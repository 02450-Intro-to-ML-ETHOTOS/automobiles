# goal: estimate the generalization error for different values of lambda
# use algorithm 5 from the book, i.e.: K-fold cross-validation for model selection

from regression_transform_data import *

import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, subplot, show, grid)

# set up cross validation
K = 10
# we set random_state to ensure reproducibility
CV = model_selection.KFold(K, shuffle=True, random_state=42)

# goal: choose a reasonable range of values of lambda
# set up regularization parameters # TODO: play with these values
lambdas = np.logspace(-4, 2, 16)
print("Lambdas:", lambdas)
# lambdas = np.logspace(-8, 2, 10)
# lambdas = np.logspace(-8, 2, 2)
S = len(lambdas)

ws = np.empty((K, S, M))
train_error = np.empty((K, S))
test_error = np.empty((K, S))

# for each value [of lambda] use K = 10 fold cross-validation to estimate the
# generalization error

# do cross-validation steps
# Iterate over k=1,...,K splits
k = 0
for train_index, test_index in CV.split(X, y):

    # Let Dk^train, Dk^test be the k'th split of D
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # TODO: batch normalize data here instead of during transform?
    # this is tricky, since we should not do this for one-hot encoded attrs

    # Iterate over s=1,...,S models
    # precompute xTx and xTy for
    # w* = (xTx + λI)^{-1} . xTy
    # recall, @ is the dot-product operator
    xTx = X_train.T @ X_train
    xTy = X_train.T @ y_train
    for s in range(0, S):
        # Train model Ms on the data Dk^train

        # λI
        lamb = lambdas[s]
        lambdaI = lamb * np.eye(M)
        # TODO: but M is number of attributes, so should be M+1?
        lambdaI[0, 0] = 0  # remove bias regularization

        # Calculate w*
        # TOS: in the exercises, they use linalg solve, but why do that,
        # when we have an analytical soluton?
        # The two following approaches are equivalent, but results differ due
        # to numerical imprecision
        w_star = np.linalg.solve(xTx+lambdaI, xTy).squeeze()
        # w = np.linalg.inv(xTx + lambdaI) @ xTy

        # TOS: note that division is not defined for matrices, e.g. A/B
        # but is equivalent to taking the inverse of matrix A and then
        # the dot product of A^-1 and B, i.e. A^-1 . B
        # w* = (xTx + λI)^{-1} . xTy
        # w_star = (np.linalg.inv(xTx + lambdaI) @ xTy).reshape((-1, 1))

        # predict
        y_train_pred = (X_train @ w_star).squeeze()  # squeeze to get 1D vector
        y_test_pred = (X_test @ w_star).squeeze()

        # print(y_train[:3], y_train_pred[:3]) # uncomment to compare y and y_pred
        # print(y_test[:3], y_test_pred[:3])  # uncomment to compare y and y_pred

        # Calculate error
        # i.e. squared loss + complexity penalty controlled by λ
        # Many equivalent ways:
        # L' = L + λ||w||^2
        # L' = ||y - y_pred||^2 + λ||w||^2
        # L' = ||y - (wTx)||^2 + λ(wTw)
        # L' = (y - y_pred)T(y - y_pred) + λ(wTw)
        # we could in fact skip the penalty term, as the sq error still depends on lambda
        wTw = w_star.T @ w_star

        y_delta_train = y_train - y_train_pred
        # divide by N observations to get the mean for the fold
        error_train = np.power(y_train-X_train @ w_star.T, 2).mean(axis=0)
        # error_train = (y_delta_train.T @ y_delta_train)  # + lamb * wTw

        y_delta_test = y_test - y_test_pred
        error_test = (y_delta_test.T @ y_delta_test)/N  # + lamb * wTw

        # store results
        # recall: k=fold, s=model
        ws[k, s, :]
        train_error[k, s] = error_train
        test_error[k, s] = error_test

    k += 1


# finally, choose optimal lambda
# min of mean validation error over K folds
opt_val_err = np.min(np.mean(test_error, axis=0))
# the lambda that gives opt_val_err
opt_lambda_idx = np.argmin(np.mean(test_error, axis=0))
opt_lambda = lambdas[opt_lambda_idx]

# calculate mean error over k folds for each lambda
train_err_vs_lambda = np.mean(train_error, axis=0)
test_err_vs_lambda = np.mean(test_error, axis=0)

# mean coefficient for plotting
# mean over folds yields matrix of shape (S,M), i.e. for each lambda term, a weight w
mean_w_vs_lambda = np.squeeze(np.mean(ws, axis=0))

# Let E_Ms,k^test be the test error of the model Ms when it is tested on Dk^test
# For each s compute E_Ms^gen = ∑^K N_k^test/N * E_Ms,k^test

# Select the optimal model, s* = argmin(s) E_Ms^gen

print('Regularized linear regression:')
print('Performance with optimal lambda: {0}'.format(opt_lambda))
print('- CV Training error (squared): {0}'.format(train_err_vs_lambda[opt_lambda_idx]))
print('- CV Test error (squared):     {0}'.format(test_err_vs_lambda[opt_lambda_idx]))


# plots
figure(k, figsize=(12, 8))
subplot(1, 2, 1)
semilogx(lambdas, mean_w_vs_lambda[:, 1:], '.-')  # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
# You can choose to display the legend, but it's omitted for a cleaner
# plot, since there are many attributes
legend(range(0, M), loc='best')

subplot(1, 2, 2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas, train_err_vs_lambda.T, 'b.-',
       lambdas, test_err_vs_lambda.T, 'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error', 'Validation error'])
grid()

show()
