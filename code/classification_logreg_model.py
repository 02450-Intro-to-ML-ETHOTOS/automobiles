from classification_transform_data import *
# imports: numpy as np, pandas as pd
import sklearn.linear_model as lm

def onehot2classidx(y):
    classes = [np.argmax(enc) for enc in y]
    return np.array(classes)

class ClassificationLogisticRegressionModel(object):
    """A regularized logistic regression model"""
    def __init__(self):
        self.mdl = None
        self.lambda_opt = None

    def fit(self, X, y, lambdas, K):
        if len(y.shape) != 1:
            # sklearn linear_model expects no one-hot encoding
            y = onehot2classidx(y)

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
            # print(f"CV Fold: {k+1}/{K}")

            # Let Dk^train, Dk^test be the k'th split of D
            # extract training and test set for current CV fold
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

            for s, lamb in enumerate(lambdas):
                # define model with appropriate regularization strength
                mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                            tol=1e-4, random_state=42, 
                            penalty='l2', C=1/lamb,
                            max_iter=2000)

                mdl.fit(X_train, y_train)
                
                y_train_pred = mdl.predict(X_train)
                y_test_pred = mdl.predict(X_test)

                error_train = np.sum(y_train_pred!=y_train) / len(y_train)
                error_test = np.sum(y_test_pred!=y_test) / len(y_test)
                
                train_error[k, s] = error_train
                test_error[k, s] = error_test

        # calculate mean error over k folds for each lambda
        error_train_mean = np.mean(train_error, axis=0)
        error_test_mean = np.mean(test_error, axis=0)

        # finally, choose optimal lambda
        # min of mean validation error over K folds is optimal
        opt_val_err = np.min(error_test_mean)
        # the lambda that gives opt_val_err
        opt_lambda_idx = np.argmin(error_test_mean)
        opt_lambda = lambdas[opt_lambda_idx]

        # Let E_Ms,k^test be the test error of the model Ms when it is tested on Dk^test
        # For each s compute E_Ms^gen = ∑^K N_k^test/N * E_Ms,k^test

        # Select the optimal model, s* = argmin(s) E_Ms^gen

        # fit model again, using optimal lambda
        # this is what is done in ex8_1_1.py
        # define model with appropriate regularization strength
        mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                    tol=1e-4, random_state=42, 
                    penalty='l2', C=1/opt_lambda,
                    max_iter=2000)
        mdl.fit(X, y) # N.B. we train with ALL data

        self.mdl = mdl
        self.lambda_opt = opt_lambda

        return opt_lambda, opt_lambda_idx, error_train_mean, error_test_mean

    def predict(self, X):
        assert(self.mdl is not None), "Model not trained yet!"
        
        return self.mdl.predict(X)

# necessary for sklearn LogisticClassifier

# test      
# y = onehot2classidx(y)
# lambdas = np.logspace(-2, 2, 4)
# lrm = ClassificationLogisticRegressionModel()
# opt_lambda, opt_lambda_idx, train_err_vs_lambda, test_err_vs_lambda = lrm.fit(X, y, lambdas, 2)
# print(lrm.predict(X[:10,:]))
# print(y[:10])
