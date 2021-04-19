from regression_transform_data import *
# imports numpy as np, pandas as pd


class RidgeRegressionModel(object):
    """A ridge regression model"""
    def __init__(self):
        self.w_star = None
        self.opt_lambda = None

    def fit(self, X, y, lambdas, K):
        print("Fitting", type(self).__name__)
        
        N, M = X.shape

        # the number of models
        S = len(lambdas)

        # matrices for storing coefficients and errors
        ws = np.empty((K, S, M))
        train_error = np.empty((K, S))
        test_error = np.empty((K, S))

        # for each value [of lambda] use K = 10 fold cross-validation to estimate the
        # generalization error

        # set up cross validation
        CV = model_selection.KFold(K, shuffle=True, random_state=42)
        # do cross-validation steps
        # Iterate over k=1,...,K splits
        for k, (train_index, test_index) in enumerate(CV.split(X, y)):
            print(f"\tFit CV Fold: {k+1}/{K}")
            # print(f"CV Fold: {k+1}/{K}")

            # Let Dk^train, Dk^test be the k'th split of D
            # extract training and test set for current CV fold
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

            # TODO: we could batch normalize data here instead of during transform?
            # this is tricky, since we should not do this for one-hot encoded attrs
            # but would probably yield better performance

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
                # we skip the penalty term, as in the exercises, as the sq error still depends on lambda
                # wTw = w_star.T @ w_star

                y_delta_train = y_train - y_train_pred
                N_y_train = len(y_delta_train)
                # divide by N observations to get the mean for the fold
                # error_train = np.power(y_delta_train, 2).mean(axis=0)
                error_train = (y_delta_train.T @ y_delta_train)/N_y_train  # + lamb * wTw

                y_delta_test = y_test - y_test_pred
                N_y_test = len(y_delta_test)
                # error_test = np.power(y_test - X_test @ w_star.T, 2).mean(axis=0)
                error_test = (y_delta_test.T @ y_delta_test)/N_y_test  # + lamb * wTw


                # store results
                # recall: k=fold, s=model
                ws[k, s, :] = w_star
                train_error[k, s] = error_train
                test_error[k, s] = error_test
        
        # calculate mean error over k folds for each lambda
        error_train_mean_per_lambda = np.mean(train_error, axis=0)
        error_test_mean_per_lambda = np.mean(test_error, axis=0)

        # finally, choose optimal lambda
        # min of mean validation error over K folds is optimal
        opt_val_err = np.min(error_test_mean_per_lambda)
        # the lambda that gives opt_val_err
        opt_lambda_idx = np.argmin(error_test_mean_per_lambda)
        opt_lambda = lambdas[opt_lambda_idx]

        # mean coefficient for plotting
        # mean over folds yields matrix of shape (S,M), i.e. for each lambda term, a weight w
        mean_w_vs_lambda = np.squeeze(np.mean(ws, axis=0))

        # Let E_Ms,k^test be the test error of the model Ms when it is tested on Dk^test
        # For each s compute E_Ms^gen = ∑^K N_k^test/N * E_Ms,k^test

        # Select the optimal model, s* = argmin(s) E_Ms^gen

        # fit model again, using optimal lambda
        # this is what is done in ex8_1_1.py
        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0, 0] = 0  # remove bias regularization
        xTx = X.T @ X      # N.B. using full dataset
        xTy = X.T @ y
        w_star = np.linalg.solve(xTx+lambdaI, xTy).squeeze()
        # w_star = (np.linalg.inv(xTx + lambdaI) @ xTy).squeeze()

        self.w_star = w_star
        self.lambda_opt = opt_lambda
        
        return opt_lambda, opt_lambda_idx, error_train_mean_per_lambda, error_test_mean_per_lambda, mean_w_vs_lambda

    def predict(self, X):
        assert(self.w_star is not None), "Model not trained yet!"

        y_pred = (X @ self.w_star).squeeze()
        return y_pred

# uncomment to test
# lambdas = np.logspace(-2, 2, 32)
# rr_model = RidgeRegressionModel()
# opt_lambda, opt_lambda_idx, train_err_vs_lambda, test_err_vs_lambda, mean_w_vs_lambda = rr_model.fit(X, y, lambdas, 10)
# print(rr_model.predict(X))

