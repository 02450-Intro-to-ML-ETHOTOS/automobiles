# goal: estimate the generalization error for different values of lambda
# use algorithm 5 from the book, i.e.: K-fold cross-validation for model selection

from regression_transform_data import *
# imports numpy as np, pandas as pd

from regression_regularized_model import *

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, subplot, show, grid)


# goal: choose a reasonable range of values of lambda
# set up regularization parameters # TODO: play with these values
lambdas = np.logspace(-2, 2, 32)
# lambdas = np.logspace(-8, 2, 10)
# lambdas = np.logspace(-8, 2, 2)
print("Lambdas:", lambdas)

rr_model = RidgeRegressionModel()
opt_lambda, opt_lambda_idx, train_err_vs_lambda, test_err_vs_lambda, mean_w_vs_lambda = rr_model.fit(X, y, lambdas, 10)

print('Regularized linear regression:')
print('Performance with optimal lambda: {0}'.format(opt_lambda))
print('- CV Training error (squared): {0}'.format(train_err_vs_lambda[opt_lambda_idx]))
print('- CV Test error (squared):     {0}'.format(test_err_vs_lambda[opt_lambda_idx]))


# plots
figure(figsize=(12, 8))
subplot(1, 2, 1)
semilogx(lambdas, mean_w_vs_lambda[:, 1:], '.-')  # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
# You can choose to display the legend, but it's omitted for a cleaner
# plot, since there are many attributes
# legend(range(0, M), loc='best') # uncomment to enable legend - though it does not fit on screen

subplot(1, 2, 2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas, train_err_vs_lambda.T, 'b.-',
       lambdas, test_err_vs_lambda.T, 'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error', 'Validation error'])
grid()

show()
